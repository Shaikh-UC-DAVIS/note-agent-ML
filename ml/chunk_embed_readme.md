# 1. Overview

1. **Chunking** — Convert raw text into token-based sliding-window chunks.
2. **Embedding** — Convert each chunk into a dense vector using RoBERTa.

```
Raw Text
   ↓
Tokenization (RoBERTa tokenizer)
   ↓
Sliding Window Chunking
   ↓
chunks.jsonl
   ↓
RoBERTa Encoder
   ↓
embeddings.npy
```

* Chunking is performed on **model tokens**, not characters.
* Embedding uses **mean pooling over last_hidden_state**.
* Output vectors are suitable for indexing (e.g., FAISS, Elasticsearch, pgvector).

---

# 2. Chunking

## Script

```
chunk_text.py
```

## Behavior

* Tokenizer: `roberta-base`
* Default window size: `20` tokens
* Default overlap: `5` tokens
* Sliding window step = `window_size - overlap`

Chunk boundaries are defined in token index space.

---

## Output Format — `chunks.jsonl`

Each line is a JSON object with the following schema:

```json
{
  "chunk_id": 0,
  "start_token": 0,
  "end_token": 20,
  "token_count": 20,
  "text": "The earth is round. This is a claim supported..."
}
```

### Field Definitions

| Field         | Type   | Description                                      |
| ------------- | ------ | ------------------------------------------------ |
| `chunk_id`    | int    | Sequential chunk index (0-based)                 |
| `start_token` | int    | Inclusive token index in original token sequence |
| `end_token`   | int    | Exclusive token index                            |
| `token_count` | int    | Number of tokens in this chunk                   |
| `text`        | string | Decoded chunk text (special tokens removed)      |

### Important Notes

* `start_token` / `end_token` allow deterministic reconstruction.
* Token indices correspond exactly to RoBERTa tokenizer output.
* Chunking must use the **same tokenizer as embedding** to preserve alignment.

---

# 3. Embedding

## Script

```
embed_roberta.py
```

## Model

* Default: `roberta-base`
* Encoder-only model
* No classification head

---

## Embedding Method

For each chunk:

1. Re-tokenize chunk text
2. Forward pass through RoBERTa
3. Extract `last_hidden_state` → shape `[B, T, H]`
4. Apply **mean pooling over tokens** using attention mask:

[
embedding = \frac{\sum_{t} h_t \cdot mask_t}{\sum mask_t}
]

This produces one vector per chunk.

---

## Output Files

### 1. `output_test_embedded_meta.json`

* Shape: `[num_chunks, hidden_dim]`
* Type: `float32`
* For `roberta-base`, `hidden_dim = 768`

Example:

```
(3, 768)
```

---

### 2.  `output_test_embedded.npy`

* You can use [NPY File Viewer](https://perchance.org/npy-file-viewer) to view the npy file (embedded vectors) online for free. 

```json
{
  "model_name": "roberta-base",
  "num_chunks": 3,
  "hidden_dim": 768,
  "pooling": "mean_pool"
}
```

---

# 4. Consistency Requirements

To ensure correctness:

* Use the same `model_name` for chunking and embedding.
* Do not modify tokenizer settings between steps.
* Keep token window parameters stable for reproducibility.

---

# 5. Execution Example

Have dependencies ready:

```
pip install transformers torch numpy
```

Chunking:

```
bash chunk.sh
```

Embedding:

```
bash embed.sh
```

