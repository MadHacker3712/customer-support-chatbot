"""
Step 5 of 5 — FastAPI chatbot endpoint.
Run AFTER quantize_and_benchmark.py.
Loads the fine-tuned model, applies int8 quantization, exposes:
  GET  /health  — liveness check
  POST /chat    — {"message": "..."} → {"response": "...", "latency_ms": 123.4}

Run locally:  python app.py
Then test:    curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d "{\"message\": \"I need to cancel my order\"}"
"""

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_DIR = os.getenv("MODEL_DIR", "./models/chatbot-finetuned")
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "80"))

# ── Structured logging ────────────────────────────────────────────────────────
_LOG_DIR = Path(__file__).parent / "logs"
_LOG_DIR.mkdir(exist_ok=True)
_LOG_FILE = _LOG_DIR / "chatbot.jsonl"


def _log_inference(input_len: int, output_len: int, latency_ms: float,
                   status: str, error: str = None) -> None:
    """Append one JSON line to logs/chatbot.jsonl."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "service": "chatbot",
        "event": "inference",
        "latency_ms": round(latency_ms, 2),
        "input_length": input_len,
        "output_length": output_len,
        "model": "DialoGPT-small-int8",
        "status": status,
        "error": error,
        "estimated_cost_usd": round((input_len + output_len) / 4 / 1000 * 0.001, 8),
    }
    with open(_LOG_FILE, "a", encoding="utf-8") as _f:
        _f.write(json.dumps(entry) + "\n")

app = FastAPI(
    title="Customer Support Chatbot API",
    description="Fine-tuned DialoGPT-small with int8 quantization",
    version="1.0.0",
)

tokenizer = None
model = None


@app.on_event("startup")
def load_model():
    global tokenizer, model
    if not os.path.exists(MODEL_DIR):
        logger.error(f"Model directory not found: {MODEL_DIR}")
        logger.error("Run finetune.py first.")
        raise RuntimeError(f"Model not found at {MODEL_DIR}")

    logger.info(f"Loading model from {MODEL_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)

    # Apply int8 quantization for faster inference
    model = torch.quantization.quantize_dynamic(
        base_model,
        {torch.nn.Linear},
        dtype=torch.qint8,
    )
    model.eval()
    logger.info("Model loaded and quantized (int8). Ready.")


# ── Request / Response schemas ────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str
    latency_ms: float


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    input_text = f"Customer: {message}\nAgent:"
    inputs = tokenizer(input_text, return_tensors="pt")

    t0 = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    latency_ms = (time.perf_counter() - t0) * 1000

    # Decode only the newly generated tokens (skip the input prompt)
    new_token_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    response_text = tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()

    logger.info(f"Query: '{message[:60]}' | Latency: {latency_ms:.1f}ms")
    _log_inference(
        input_len=inputs["input_ids"].shape[1],
        output_len=len(new_token_ids),
        latency_ms=latency_ms,
        status="ok",
    )

    return ChatResponse(response=response_text, latency_ms=round(latency_ms, 1))


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
