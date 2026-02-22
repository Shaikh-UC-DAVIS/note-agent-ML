#!/usr/bin/env python3
"""
chunk_text.py

Sliding-window chunking over tokenized text.

Default:
- window_size = 20 tokens
- overlap = 5 tokens
- tokenizer = roberta-base

Output: JSONL file with one chunk per line.
"""

import argparse
import json
from pathlib import Path
from typing import List
from transformers import AutoTokenizer


def sliding_window_ranges(n_tokens: int, window_size: int, overlap: int):
    if window_size <= 0:
        raise ValueError("window_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= window_size:
        raise ValueError("overlap must be < window_size")

    step = window_size - overlap
    ranges = []
    start = 0

    while start < n_tokens:
        end = min(start + window_size, n_tokens)
        ranges.append((start, end))
        if end == n_tokens:
            break
        start += step

    return ranges


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_text_file", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--model_name", default="roberta-base")
    parser.add_argument("--window_size", type=int, default=20)
    parser.add_argument("--overlap", type=int, default=5)
    args = parser.parse_args()

    text = Path(args.input_text_file).read_text(encoding="utf-8").strip()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    encoded = tokenizer(
        text,
        add_special_tokens=True,
        return_tensors=None,
        truncation=False,
    )

    token_ids = encoded["input_ids"]
    n_tokens = len(token_ids)

    ranges = sliding_window_ranges(n_tokens, args.window_size, args.overlap)

    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for i, (start, end) in enumerate(ranges):
            chunk_ids = token_ids[start:end]
            chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True).strip()

            record = {
                "chunk_id": i,
                "start_token": start,
                "end_token": end,
                "token_count": len(chunk_ids),
                "text": chunk_text,
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Generated {len(ranges)} chunks.")
    print(f"Output written to {args.output_jsonl}")


if __name__ == "__main__":
    main()
