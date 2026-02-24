#!/usr/bin/env bash
set -euo pipefail

# Always run relative to this script's directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Config (edit if needed)
MODEL_NAME="roberta-base"
BATCH_SIZE=16
MAX_LENGTH=256

CHUNKS_JSONL="chunks.jsonl"
OUTPUT_DIR="$SCRIPT_DIR"  # Save embeddings in the same dir as this script

echo "=== Embedding started: $(date) ==="

if [[ ! -f "$CHUNKS_JSONL" ]]; then
  echo "ERROR: Missing chunks file: ${CHUNKS_JSONL}"
  echo "Run ./run_chunking.sh first."
  exit 1
fi

if [[ ! -f "embed_roberta.py" ]]; then
  echo "ERROR: Missing embed_roberta.py in ${SCRIPT_DIR}"
  exit 1
fi

python3 embed_roberta.py \
  --chunks_jsonl "$CHUNKS_JSONL" \
  --output_dir "$OUTPUT_DIR" \
  --model_name "$MODEL_NAME" \
  --batch_size "$BATCH_SIZE" \
  --max_length "$MAX_LENGTH"

echo
echo "=== Embedding finished: $(date) ==="
echo "Embeddings written to: $OUTPUT_DIR"
