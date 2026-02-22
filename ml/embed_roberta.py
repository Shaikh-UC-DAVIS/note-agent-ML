#!/usr/bin/env python3
"""
embed_roberta.py

Generates embeddings for chunks using RoBERTa.
Uses mean pooling over last_hidden_state.

Input: JSONL file with one chunk per line gained from chunk_test.py
"""

import argparse
import json
from pathlib import Path
from typing import List
import numpy as np
import torch
from transformers import AutoTokenizer, RobertaModel
import warnings
import logging

warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked_embeddings = last_hidden_state * mask
    summed = masked_embeddings.sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def load_chunks(path: Path):
    chunks = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name", default="roberta-base")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    chunks = load_chunks(Path(args.chunks_jsonl))
    texts = [c["text"] for c in chunks]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = RobertaModel.from_pretrained(args.model_name).to(device)
    model.eval()

    embeddings = []

    with torch.no_grad():
        for i in range(0, len(texts), args.batch_size):
            batch_texts = texts[i:i + args.batch_size]

            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt"
            )

            encoded = {k: v.to(device) for k, v in encoded.items()}

            outputs = model(**encoded)
            pooled = mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
            embeddings.append(pooled.cpu().numpy())

    embeddings = np.vstack(embeddings)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    np.save(output_dir / "output_test_embedded.npy", embeddings)

    metadata = {
        "model_name": args.model_name,
        "num_chunks": embeddings.shape[0],
        "hidden_dim": embeddings.shape[1],
        "pooling": "mean_pool",
    }

    (output_dir / "output_test_embedded_meta.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8"
    )

    print("Embeddings shape:", embeddings.shape)
    print("Saved to", output_dir)


if __name__ == "__main__":
    main()
