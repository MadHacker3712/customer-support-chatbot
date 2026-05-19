"""
Step 2 of 5 — Fine-tune DialoGPT-small on 500 customer support examples.
Run AFTER download_dataset.py.
Takes 30-90 minutes on CPU. Go do Khan Academy while this runs.
Saves the model to models/chatbot-finetuned/
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)

MODEL_NAME = "microsoft/DialoGPT-small"
OUTPUT_DIR = "./models/chatbot-finetuned"
DATA_PATH = "./data/train_subset.csv"
MAX_LENGTH = 128


class CustomerSupportDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_length: int = 128):
        df = pd.read_csv(data_path)
        self.encodings = []
        for _, row in df.iterrows():
            # Format: "Customer: {question}\nAgent: {answer}<|endoftext|>"
            text = (
                f"Customer: {row['instruction']}\n"
                f"Agent: {row['response']}"
                f"{tokenizer.eos_token}"
            )
            enc = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            self.encodings.append({k: v.squeeze(0) for k, v in enc.items()})

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        item = dict(self.encodings[idx])
        item["labels"] = item["input_ids"].clone()
        return item


def main():
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: {DATA_PATH} not found. Run download_dataset.py first.")
        return

    print("=" * 50)
    print(f"Loading model: {MODEL_NAME}")
    print("=" * 50)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    dataset = CustomerSupportDataset(DATA_PATH, tokenizer, MAX_LENGTH)
    print(f"Training examples: {len(dataset)}")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        warmup_steps=50,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=25,
        save_strategy="epoch",
        no_cuda=True,          # CPU training
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    print("\nStarting fine-tuning (30-90 min on CPU)...")
    print("You'll see loss printed every 25 steps. It should decrease over time.\n")
    trainer.train()

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nModel saved to {OUTPUT_DIR}")
    print("Next: run  python benchmark.py")


if __name__ == "__main__":
    main()
