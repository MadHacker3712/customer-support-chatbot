"""
Step 3 of 5 — Baseline benchmark (fp32, no quantization).
Run AFTER finetune.py finishes.
Measures: model size (MB), avg latency (ms), peak memory (MB).
Saves results to results/baseline_metrics.json
"""

import json
import os
import sys
import time

import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = "./models/chatbot-finetuned"
MAX_NEW_TOKENS = 80
RESULTS_DIR = "./results"

TEST_PROMPTS = [
    "I need to cancel my order",
    "Where is my package?",
    "I want a refund for my purchase",
    "How do I change my shipping address?",
    "My order arrived damaged",
    "I was charged twice for my order",
    "How do I track my shipment?",
    "I need to return an item",
    "What is your return policy?",
    "I never received my order",
    "Can I change the size of my order?",
    "I forgot my account password",
    "How do I update my payment method?",
    "My discount code is not working",
    "I want to upgrade my subscription",
    "How do I delete my account?",
    "I got the wrong item in my order",
    "When will my order ship?",
    "Can I add items to my existing order?",
    "I need help with my invoice",
]


def get_model_size_mb(model) -> float:
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return total_bytes / (1024 ** 2)


def run_benchmark(model, tokenizer, label: str) -> dict:
    print(f"\n--- {label} ---")
    size_mb = get_model_size_mb(model)
    print(f"Model size (in-memory): {size_mb:.2f} MB")

    latencies = []
    process = psutil.Process(os.getpid())
    peak_memory = 0.0

    model.eval()
    with torch.no_grad():
        for i, prompt in enumerate(TEST_PROMPTS):
            input_text = f"Customer: {prompt}\nAgent:"
            inputs = tokenizer(input_text, return_tensors="pt")

            t0 = time.perf_counter()
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            t1 = time.perf_counter()

            mem_mb = process.memory_info().rss / (1024 ** 2)
            peak_memory = max(peak_memory, mem_mb)
            latencies.append((t1 - t0) * 1000)
            print(f"  [{i+1:02d}/{len(TEST_PROMPTS)}] {(t1-t0)*1000:.0f} ms", end="\r")

    avg_latency = sum(latencies) / len(latencies)
    print(f"\nAvg latency:   {avg_latency:.1f} ms")
    print(f"Peak memory:   {peak_memory:.1f} MB")

    return {
        "label": label,
        "size_mb": round(size_mb, 2),
        "avg_latency_ms": round(avg_latency, 2),
        "peak_memory_mb": round(peak_memory, 1),
    }


def main():
    if not os.path.exists(MODEL_DIR):
        print(f"ERROR: {MODEL_DIR} not found. Run finetune.py first.")
        sys.exit(1)

    print("=" * 50)
    print("BASELINE BENCHMARK (fp32)")
    print("=" * 50)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)

    results = run_benchmark(model, tokenizer, "Baseline (fp32)")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "baseline_metrics.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved → {out_path}")
    print("Next: run  python quantize_and_benchmark.py")


if __name__ == "__main__":
    main()
