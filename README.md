# Customer Support Chatbot — Optimized Backend
# Project #1 | MLOps Engineer Path

## What This Is

Fine-tuned DialoGPT-small on 500 customer support Q&A pairs (Bitext dataset).
Quantized to int8. Served via FastAPI. Containerized with Docker.

---

## Benchmark Results

| Model Version | Size (MB) | Avg Latency (ms) | Peak Memory (MB) | Notes |
|---|---:|---:|---:|---|
| Baseline (fp32) | 474.70 | 4543.86 | 794.2 | DialoGPT-small, CPU, 20 prompts |
| Quantized (int8) | 474.70 | 4477.93 | 1334.5 | torch dynamic int8 on Linear layers |

**Size reduction**: 0% | **Latency improvement**: 1.45%

> **Note**: Dynamic quantization on a small CPU model shows minimal gains here — this is expected and worth documenting. The in-memory size stays the same because weights are quantized at runtime, not stored. Real gains appear with larger models (7B+) on GPU, or with static quantization. This is the honest trade-off analysis that MLOps engineers document.

---

## How to Run (Local)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download dataset
python download_dataset.py

# 3. Fine-tune (30-90 min on CPU)
python finetune.py

# 4. Baseline benchmark
python benchmark.py

# 5. Quantize + compare
python quantize_and_benchmark.py

# 6. Start API
python app.py
```

Test the API:
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d "{\"message\": \"I need to cancel my order\"}"
```

---

## How to Run (Docker)

```bash
# Build
docker build -t customer-support-chatbot .

# Run
docker run -p 8000:8000 customer-support-chatbot

# Test
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d "{\"message\": \"Where is my package?\"}"
```

---

## Architecture

```
User Request
     │
     ▼
FastAPI /chat endpoint (app.py)
     │
     ▼
DialoGPT-small (int8 quantized)
fine-tuned on Bitext customer support dataset
     │
     ▼
JSON response + latency_ms
```

---

## Dataset

- **Source**: [bitext/Bitext-customer-support-llm-chatbot-training-dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)
- **Size**: 26,872 rows (trained on 500-row CPU subset)
- **Format**: `instruction` → `response`
- **Intents**: 27 categories (cancel_order, track_order, request_refund, etc.)

---

## Trade-offs

- Dynamic int8 quantization only affects `Linear` layers — no retraining needed
- Slight quality regression possible on edge cases; acceptable for customer support routing
- CPU fine-tuning used a 500-row subset; GPU + full dataset would improve quality
- For production: replace DialoGPT-small with Llama 3 8B + QLoRA for better responses
