# How I Containerized an LLM: A Practical MLOps Guide

*By MadHacker3712 | May 2026*

---

Most AI tutorials end at "run this script locally." That's not production. In production, your model needs to run the same way on your laptop, a teammate's machine, and a cloud server—without anyone asking you "wait, which Python version?" That's what Docker solves.

This is how I containerized a fine-tuned customer support chatbot end to end.

---

## What I Built

A FastAPI server that wraps a fine-tuned DialoGPT-small model, quantized to int8 for faster inference. The API accepts a customer question and returns an agent-style response:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "I need to cancel my order"}'

# Response:
# {"response": "I understand you'd like to cancel your order...", "latency_ms": 312.4}
```

One command. Works anywhere Docker runs.

---

## Why Docker for LLMs?

When you ship an LLM-backed service without Docker, you're shipping a list of instructions. When you ship it *with* Docker, you're shipping the environment itself.

Three specific problems Docker solves for LLM work:

1. **Dependency hell**: PyTorch, transformers, tokenizers—these all have version-sensitive interactions. Docker freezes the exact versions that worked.
2. **Reproducibility**: Your benchmark numbers only mean something if someone else can reproduce them. A containerized model is reproducible by definition.
3. **Deployment parity**: The same image runs locally and in the cloud. No surprises in production.

---

## The Dockerfile

Here's the full Dockerfile I wrote, with every decision explained:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Copy requirements first — Docker caches this layer.
# If only app.py changes, pip install does NOT re-run.
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Copy fine-tuned model
COPY models/chatbot-finetuned ./models/chatbot-finetuned

# Copy application code
COPY app.py .

EXPOSE 8000

ENV MODEL_DIR=./models/chatbot-finetuned
ENV MAX_NEW_TOKENS=80

CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Key decisions:**

- **`python:3.11-slim`** — not the full Python image. Slim removes docs, tests, and package manager caches. Keeps the image smaller.
- **Requirements before code** — this is the most important layer caching trick. Docker builds layers in order and caches each one. If `requirements.txt` hasn't changed, that entire 2-minute pip install gets skipped on rebuild. Only the changed code layer re-runs.
- **CPU-only torch** — the `--index-url` flag points pip at PyTorch's CPU wheel instead of the default GPU version. This cuts 1.5GB from the image. For CPU inference, you don't need CUDA.
- **ENV for config** — `MODEL_DIR` and `MAX_NEW_TOKENS` as environment variables means you can override them at runtime without rebuilding the image.

---

## What I Learned About Layer Caching

Docker images are built in layers. Each instruction (`FROM`, `COPY`, `RUN`) creates a new layer. Docker caches each layer and only rebuilds from the first changed layer downward.

This is why order matters:

```dockerfile
# WRONG — changes to app.py re-run pip install
COPY . .
RUN pip install -r requirements.txt

# RIGHT — changes to app.py only re-copy app.py
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY app.py .
```

The right approach cut my rebuild time from ~8 minutes to ~10 seconds when I only changed `app.py`.

---

## The Quantization Inside the Container

The model loads as fp32, then gets dynamically quantized to int8 at startup:

```python
model = torch.quantization.quantize_dynamic(
    base_model,
    {torch.nn.Linear},
    dtype=torch.qint8,
)
```

This happens every time the container starts. It takes about 2 seconds and means I don't have to store a separate quantized model file—the container always starts from the clean fine-tuned weights and quantizes on the fly.

**Benchmark results on CPU:**

| Version | Size | Avg Latency | Peak Memory |
|---|---:|---:|---:|
| Baseline (fp32) | 474.70 MB | 4543.86 ms | 794.2 MB |
| Quantized (int8) | 474.70 MB | 4477.93 ms | 1334.5 MB |

Only 1.45% latency improvement. I documented this honestly: dynamic quantization on a small CPU model shows minimal gains because the quantization overhead offsets the speedup. On large models (7B+) on GPU, this same technique shows 30-50% latency reduction. That's the trade-off analysis that matters in production—not just "did it get faster" but "why, and when does it not."

---

## Build, Run, Push

```bash
# Build
docker build -t topdeveloper123/customer-support-chatbot:v1 .

# Test locally
docker run -p 8000:8000 topdeveloper123/customer-support-chatbot:v1

# Push to Docker Hub (anyone can now pull and run your model)
docker push topdeveloper123/customer-support-chatbot:v1
```

Anyone with Docker installed can now run your model with one command:

```bash
docker run -p 8000:8000 topdeveloper123/customer-support-chatbot:v1
```

No Python setup. No pip install. No version conflicts. That's the point.

---

## What's Next

This is Project 1 of my MLOps learning path. Next I'll be:
- Comparing inference setups: standard pipeline vs quantized vs vLLM
- Building a RAG backend with vector search optimization
- Adding production monitoring and latency alerting

GitHub: [github.com/MadHacker3712/customer-support-chatbot](https://github.com/MadHacker3712/customer-support-chatbot)

---

*Learning MLOps in public. Building real systems, not tutorials.*
