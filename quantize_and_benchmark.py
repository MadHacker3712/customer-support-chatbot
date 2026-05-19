"""
Step 4 of 5 — Quantize to int8 and benchmark.
Run AFTER benchmark.py.
Applies torch dynamic int8 quantization, re-runs the same 20 prompts,
prints a comparison table, and saves results/comparison.json
"""

import json
import os
import sys
import time

import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = "./models/chatbot-finetuned"
QUANTIZED_DIR = "./models/chatbot-quantized"
RESULTS_DIR = "./results"
MAX_NEW_TOKENS = 80

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


def benchmark(model, tokenizer, label: str) -> dict:
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

    baseline_path = os.path.join(RESULTS_DIR, "baseline_metrics.json")
    if not os.path.exists(baseline_path):
        print("WARNING: results/baseline_metrics.json not found.")
        print("Run benchmark.py first so you have a comparison.")
        baseline = None
    else:
        with open(baseline_path) as f:
            baseline = json.load(f)

    print("=" * 50)
    print("QUANTIZATION BENCHMARK (int8)")
    print("=" * 50)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)

    print("\nApplying dynamic int8 quantization to Linear layers...")
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8,
    )
    print("Quantization applied.")

    results = benchmark(quantized_model, tokenizer, "Quantized (int8)")

    # Save quantized model weights + tokenizer
    os.makedirs(QUANTIZED_DIR, exist_ok=True)
    torch.save(quantized_model.state_dict(), os.path.join(QUANTIZED_DIR, "quantized_state_dict.pt"))
    tokenizer.save_pretrained(QUANTIZED_DIR)
    print(f"\nQuantized model saved → {QUANTIZED_DIR}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "quantized_metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    if baseline:
        size_reduction = ((baseline["size_mb"] - results["size_mb"]) / baseline["size_mb"]) * 100
        latency_improvement = ((baseline["avg_latency_ms"] - results["avg_latency_ms"]) / baseline["avg_latency_ms"]) * 100

        print("\n" + "=" * 56)
        print("  COMPARISON TABLE")
        print("=" * 56)
        print(f"  {'Metric':<24} {'Baseline':>10} {'Int8':>10} {'Change':>10}")
        print("  " + "-" * 52)
        print(f"  {'Size (MB)':<24} {baseline['size_mb']:>10.2f} {results['size_mb']:>10.2f} {size_reduction:>+9.1f}%")
        print(f"  {'Avg Latency (ms)':<24} {baseline['avg_latency_ms']:>10.1f} {results['avg_latency_ms']:>10.1f} {latency_improvement:>+9.1f}%")
        print(f"  {'Peak Memory (MB)':<24} {baseline['peak_memory_mb']:>10.1f} {results['peak_memory_mb']:>10.1f}")
        print("=" * 56)

        comparison = {
            "baseline": baseline,
            "quantized": results,
            "size_reduction_pct": round(size_reduction, 2),
            "latency_improvement_pct": round(latency_improvement, 2),
        }
        with open(os.path.join(RESULTS_DIR, "comparison.json"), "w") as f:
            json.dump(comparison, f, indent=2)

        print(f"\nSaved → results/comparison.json")
        print("\nCopy those numbers into the README.md benchmark table.")

    print("Next: run  python app.py   (then test with curl or Postman)")


if __name__ == "__main__":
    main()
