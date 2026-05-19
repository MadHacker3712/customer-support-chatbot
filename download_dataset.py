"""
Step 1 of 5 — Download the Bitext Customer Support dataset.
Run this first. Needs internet. Takes ~1 minute.
Saves a 500-row subset to data/train_subset.csv for fast CPU fine-tuning.
"""

import os
import pandas as pd
from datasets import load_dataset


def main():
    print("=" * 50)
    print("Downloading Bitext Customer Support dataset...")
    print("=" * 50)

    dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")
    train_data = dataset["train"]

    print(f"Total rows:  {len(train_data)}")
    print(f"Columns:     {train_data.column_names}")

    df = train_data.to_pandas()

    print("\nSample rows (instruction → response):")
    print("-" * 60)
    for _, row in df[["instruction", "response"]].head(3).iterrows():
        print(f"Customer: {row['instruction']}")
        print(f"Agent:    {row['response'][:80]}...")
        print()

    os.makedirs("data", exist_ok=True)

    # 500-row subset for CPU fine-tuning (full dataset = GPU territory)
    subset = df[["instruction", "response"]].sample(500, random_state=42)
    subset.to_csv("data/train_subset.csv", index=False)

    df[["instruction", "response"]].to_csv("data/train_full.csv", index=False)

    print(f"Saved 500-row subset  → data/train_subset.csv")
    print(f"Saved full dataset    → data/train_full.csv (26,872 rows)")
    print("\nNext: run  python finetune.py")


if __name__ == "__main__":
    main()
