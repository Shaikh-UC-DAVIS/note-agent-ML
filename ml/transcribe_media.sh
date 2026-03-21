#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /path/to/file.{mp3,wav,m4a,mp4} [note_id] [output_path]" >&2
  exit 1
fi

INPUT_PATH="$1"
NOTE_ID="${2:-}"
OUTPUT_PATH="${3:-/Users/gegekang/Desktop/note-agent-ML-chunk-embed/ml/outputs/full_transcript.txt}"

export INPUT_PATH
export NOTE_ID
export OUTPUT_PATH

python3 - <<'PY'
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(".env")

from ml.db import create_note
from ml.extraction_tasks import extract_text_task

input_path = os.environ.get("INPUT_PATH")
if not input_path:
    raise SystemExit("INPUT_PATH is not set")

note_id_env = os.environ.get("NOTE_ID")
output_path = os.environ.get("OUTPUT_PATH")

if note_id_env:
    note_id = int(note_id_env)
else:
    note_id = create_note(input_path)

text = extract_text_task(note_id)

out_path = Path(output_path)
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(text, encoding="utf-8")

print(f"note_id={note_id}")
print(f"wrote={out_path}")
print(f"chars={len(text)}")
PY
