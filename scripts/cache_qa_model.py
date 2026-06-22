#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

from transformers import AutoModelForQuestionAnswering, AutoTokenizer


MODEL_NAME = "distilbert-base-cased-distilled-squad"
OUTPUT_DIR = Path("/scratch-shared/group_h/models/distilbert-base-cased-distilled-squad")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(f"Downloading QA model: {MODEL_NAME}")
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)
    print(f"Saving tokenizer and model to: {OUTPUT_DIR}")
    tokenizer.save_pretrained(OUTPUT_DIR)
    model.save_pretrained(OUTPUT_DIR)
    print("QA model cache complete.")


if __name__ == "__main__":
    main()
