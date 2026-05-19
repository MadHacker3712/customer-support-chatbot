# Case Study: How I Cut LLM Inference Latency by 76% with Quantization and Batching

*By MadHacker3712 | May 2026*

---

## The Problem

Every time a customer sends a message to our chatbot, we run a full forward pass through a 117M-parameter language model. On CPU, that takes around 1.3–1.5 seconds per request at standard float32 precision. With a handful of concurrent users, that's fine. At production scale — dozens of simultaneous requests — it stacks up fast.

I wanted to cut latency without buying a GPU. Everything had to run on CPU.

---

## Baseline: Where I Started

Before any optimization, the setup was straightforward:
- Model: DialoGPT-small fine-tuned on a customer support dataset
- Precision: float32 (the default)
- Batch size: 1 (one request at a time)
- Average latency: **5418 ms** per batch of 4 requests (1354 ms per request)
- Throughput: **0.74 requests/second**

Numbers like these don't work in production.

---

## Optimization 1: Dynamic Int8 Quantization

The first tool I reached for was PyTorch's `quantize_dynamic`. It converts the model's linear layer weights from 32-bit floats to 8-bit integers at runtime — no retraining required.

```python
import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("./models/chatbot-finetuned")

model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},   # only quantize Linear layers
    dtype=torch.qint8,
)
```

**Why this works:**
- Weights are stored as int8 instead of float32 → **4× smaller memory footprint**
- Modern CPUs have SIMD integer instructions that run faster than floating-point equivalents
- No accuracy retraining needed — quantization happens on the fly at inference time

**Trade-off:** Activation values are still computed in float32. This is a post-training quantization technique, so there's a small quality degradation on edge cases. For a customer support chatbot generating short, factual responses, it's imperceptible.

---

## Optimization 2: Batched Inference with Left-Padding

The second optimization was batching. Instead of running 4 requests sequentially, I run them together in a single forward pass. The model's transformer layers process all sequences in parallel.

The tricky part: batch tokenization requires all sequences to be the same length. You pad shorter sequences to match the longest one. For *autoregressive generation*, padding on the left matters — the model should generate from the real tokens, not from padding.

```python
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

inputs = tokenizer(
    batch_messages,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=128,
)

with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=80,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
```

**Why left-padding specifically:** The model generates tokens from left to right. If padding is on the right, the model "sees" padding tokens in the middle of generation and produces incoherent output. Left-padding keeps all real tokens contiguous at the end of the sequence, right before generation begins.

---

## Results

I benchmarked Setup 1 (fp32, batch=1) against Setup 2 (int8, batch=4) over 5 runs of identical inputs:

| Setup | Total Latency | Latency/Request | Throughput |
|---|---:|---:|---:|
| Setup 1: fp32, batch=1 | 5418 ms | 1354 ms | 0.74 req/s |
| Setup 2: int8, batch=4 | 1277 ms | 319 ms | 3.14 req/s |
| **Improvement** | **-76.4%** | **-76.4%** | **+325%** |

**76.4% latency reduction. 4.3× throughput gain. On CPU. No hardware upgrade.**

---

## Why This Matters for Production

These numbers aren't just impressive on paper. They translate directly to cost and user experience:

- **Lower latency** → responses under 350ms feel instant to users; over 1 second feels broken
- **Higher throughput** → the same server handles 4× more users before you need to scale horizontally
- **Smaller memory** → int8 model fits in instances with less RAM (cheaper VMs)
- **No GPU required** → CPU inference is simpler to deploy (no CUDA drivers, no GPU pricing)

In a real deployment, you'd combine this with:
- A request queue (handle burst traffic without dropping requests)
- Auto-scaling (spin up more containers when queue depth exceeds threshold)
- Caching (identical queries skip inference entirely)

---

## What I'd Do Next

**GPTQ/AWQ quantization** — these are weight-only quantization methods that run on GPU. For a GPU deployment, GPTQ can deliver 2-4bit quantized models with negligible quality loss. I documented this in [vllm_example.py](vllm_example.py) alongside vLLM's PagedAttention for continuous batching.

**Flash Attention** — reduces the O(n²) memory complexity of attention layers. Not available on CPU, but critical for GPU deployments serving long contexts.

**KV Cache** — already used by default in HuggingFace `model.generate()`, but understanding it matters: the model doesn't recompute key/value pairs for previously generated tokens, which is why the second, third, and fourth tokens are generated faster than the first.

---

## Code

Full implementation: [github.com/MadHacker3712/customer-support-chatbot](https://github.com/MadHacker3712/customer-support-chatbot)  
Inference study (run_comparison.py): [github.com/MadHacker3712/mlops-inference-study](https://github.com/MadHacker3712/mlops-inference-study)

Run the benchmark yourself:
```bash
git clone https://github.com/MadHacker3712/mlops-inference-study
cd mlops-inference-study
pip install -r requirements.txt
python run_comparison.py
```
