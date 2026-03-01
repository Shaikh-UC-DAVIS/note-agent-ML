#!/usr/bin/env bash
set -euo pipefail

PDF_PATH="${1:-uploads/ws_1/1/file_1.pdf}"
WORKSPACE_ID="${2:-ws_1}"
FILE_ID="${3:-file_1}"

if [[ ! -f "$PDF_PATH" ]]; then
  echo "PDF not found: $PDF_PATH" >&2
  exit 1
fi

python3 - <<PY
import sys
from pathlib import Path

cwd = Path.cwd()
sys.path.insert(0, str(cwd.parent))

from ml.db import create_note, update_note
from ml.extraction_tasks import extract_text_task, chunk_text_task

note_id = create_note("${PDF_PATH}", status="uploaded")
update_note(note_id, workspace_id="${WORKSPACE_ID}", file_id="${FILE_ID}", mime_type="application/pdf")

extract_text_task(note_id)
chunk_text_task(note_id)

print("note_id:", note_id)
PY
