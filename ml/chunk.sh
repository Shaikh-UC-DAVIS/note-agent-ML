#!/usr/bin/env bash
set -euo pipefail

# Always run relative to this script's directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Config (edit if needed)
MODEL_NAME="roberta-base"
WINDOW_SIZE=20
OVERLAP=5
INPUT_TEXT_FILE="input_test_chunk_embed.txt"
CHUNKS_JSONL="chunks.jsonl"


  echo "=== Chunking started: $(date) ==="
  if [[ ! -f "$INPUT_TEXT_FILE" ]]; then
    echo "ERROR: Missing input file: ${INPUT_TEXT_FILE}"
    exit 1
  fi

  if [[ ! -f "chunk_text.py" ]]; then
    echo "ERROR: Missing chunk_text.py in ${SCRIPT_DIR}"
    exit 1
  fi

  python3 chunk_text.py \
    --input_text_file "$INPUT_TEXT_FILE" \
    --output_jsonl "$CHUNKS_JSONL" \
    --model_name "$MODEL_NAME" \
    --window_size "$WINDOW_SIZE" \
    --overlap "$OVERLAP"

  echo
  echo "=== Chunking finished: $(date) ==="
  echo "Chunks written to: $CHUNKS_JSONL"
