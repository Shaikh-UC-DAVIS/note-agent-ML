# 1. Overview

1. **Extraction** — Extract text from PDF/DOCX/image/text into `extracted.txt`.
2. **Chunking** — Split extracted text into 400–800 token windows with 10% overlap.
3. **Embedding** — Convert each chunk into a dense vector using RoBERTa.

```
Raw Text
   ↓
Extraction (PDF/DOCX/Image/Text)
   ↓
Sentence Tokenization (spaCy)
   ↓
Sliding Window Chunking (tiktoken)
   ↓
chunks.jsonl
   ↓
RoBERTa Encoder
   ↓
embeddings.npy
```

* Extraction is file-type aware with fallbacks (PDF → PyPDF2 → OCR).
* Chunking is performed on **model tokens**, not characters.
* Embedding uses **mean pooling over last_hidden_state**.
* Output vectors are suitable for indexing (e.g., FAISS, Elasticsearch, pgvector).

---

# 2. Extraction (Stage 1)

## What it does

* Routes by mime type or file extension.
* PDF: `pdfplumber` → `PyPDF2` → OCR via `pytesseract`.
* DOCX: `python-docx`.
* Images: OCR via `pytesseract`.
* Text: read directly (UTF-8).
* Writes: `derived/{workspace_id}/{note_id}/extracted.txt`
* Updates DB: `notes.status = 'extracted'`, stores `content_hash`.

## Run (PDF example)

```
bash pdf_uploading.sh uploads/ws_1/1/file_1.pdf ws_1 file_1
```

This creates a note, runs extraction + chunking, and prints the `note_id`.

---

# 3. Chunking (Stage 2)

## Script (pipeline task)

```
extraction_tasks.py (chunk_text_task)
```

## Behavior

* Sentence segmentation: `spaCy` (`en_core_web_sm`).
* Token counting: `tiktoken` (`cl100k_base` by default).
* Window size: 400–800 tokens (defaults 800).
* Overlap: 10% (defaults 80 tokens).
* Writes: `derived/{workspace_id}/{note_id}/chunks.jsonl`
* Updates DB: `notes.status = 'chunked'`, inserts into `spans` table.

---

## Output Format — `chunks.jsonl`

Each line is a JSON object with the following schema:

```json
{
  "chunk_id": "span_1",
  "start_char": 0,
  "end_char": 182,
  "token_count": 47,
  "position": 0,
  "text": "The earth is round. This is a claim supported..."
}
```

### Field Definitions

| Field         | Type   | Description                                      |
| ------------- | ------ | ------------------------------------------------ |
| `chunk_id`    | string | Sequential chunk id (`span_1`, `span_2`, ...)     |
| `start_char`  | int    | Inclusive character index in original document   |
| `end_char`    | int    | Exclusive character index                        |
| `token_count` | int    | Number of tokens in this chunk                   |
| `position`    | int    | 0-based position in document                     |
| `text`        | string | Chunk text                                       |

### Important Notes

* `start_char` / `end_char` allow deterministic reconstruction.
* Token counts are from `tiktoken` for window sizing only.

---

# 4. Embedding

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

# 5. Consistency Requirements

To ensure correctness:

* Keep chunking parameters stable for reproducibility.
* Use the same encoder model for downstream indexing/search.

---

# 6. Execution Example

Have dependencies ready:

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Extraction + Chunking (PDF example):

```
bash pdf_uploading.sh uploads/ws_1/1/file_1.pdf ws_1 file_1
```

Embedding:

```
bash embed.sh
```
