"""
tasks.py

Extraction and chunking pipeline tasks for notes.
"""

import json
import os
import re
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import tiktoken

from ml.db import delete_spans, get_note, insert_spans, update_note

try:
    import pdfplumber
except Exception:  # pragma: no cover - optional dependency for runtime use
    pdfplumber = None

try:
    import PyPDF2
except Exception:  # pragma: no cover - optional dependency for runtime use
    PyPDF2 = None

try:
    from docx import Document
except Exception:  # pragma: no cover - optional dependency for runtime use
    Document = None

try:
    import pytesseract
except Exception:  # pragma: no cover - optional dependency for runtime use
    pytesseract = None

try:
    from PIL import Image
except Exception:  # pragma: no cover - optional dependency for runtime use
    Image = None

try:
    import spacy
except Exception:  # pragma: no cover - optional dependency for runtime use
    spacy = None


REPO_ROOT = Path(__file__).resolve().parents[1]
UPLOADS_DIR = Path(os.environ.get("NOTE_AGENT_UPLOADS_DIR", REPO_ROOT / "uploads"))
DERIVED_DIR = Path(os.environ.get("NOTE_AGENT_DERIVED_DIR", REPO_ROOT / "derived"))


def _resolve_pdf_path(file_path: str) -> Path:
    path = Path(file_path)
    if path.is_absolute():
        return path
    return (UPLOADS_DIR / path).resolve()


def _clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ")
    # De-hyphenate line breaks: "exam-\nple" -> "example"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # Remove page number-only lines
    text = re.sub(r"^\s*(Page\s+)?\d+(\s*/\s*\d+)?\s*$", "", text, flags=re.MULTILINE)
    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _ocr_page(page) -> str:
    if pytesseract is None:
        return ""
    try:
        image = page.to_image(resolution=300).original
    except Exception:
        return ""
    try:
        return pytesseract.image_to_string(image)
    except Exception:
        return ""


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _guess_mime_type(file_path: str) -> str:
    suffix = Path(file_path).suffix.lower()
    if suffix == ".pdf":
        return "application/pdf"
    if suffix == ".docx":
        return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    if suffix in {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}:
        return "image"
    if suffix in {".txt", ".md", ".csv", ".tsv"}:
        return "text/plain"
    return "application/octet-stream"


def _resolve_file_path(note: Dict[str, Union[str, int]]) -> Path:
    file_path = note.get("file_path")
    if file_path:
        return _resolve_pdf_path(str(file_path))

    workspace_id = str(note.get("workspace_id") or "default")
    file_id = str(note.get("file_id") or note.get("id"))
    mime_type = str(note.get("mime_type") or "")
    ext = ".pdf"
    if mime_type.startswith("text/"):
        ext = ".txt"
    elif "docx" in mime_type:
        ext = ".docx"
    elif mime_type.startswith("image/") or mime_type == "image":
        ext = ".png"
    return (UPLOADS_DIR / workspace_id / str(note["id"]) / f"{file_id}{ext}").resolve()


def _derived_paths(note: Dict[str, Union[str, int]]) -> Tuple[Path, Path]:
    workspace_id = str(note.get("workspace_id") or "default")
    base_dir = DERIVED_DIR / workspace_id / str(note["id"])
    return base_dir / "extracted.txt", base_dir / "chunks.jsonl"


def _read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise UnicodeDecodeError("utf-8", b"", 0, 1, f"Invalid UTF-8 in {path}") from exc


def _extract_pdf_text_pdfplumber(pdf_path: Path) -> str:
    if pdfplumber is None:
        return ""
    pages_text: List[str] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            pages_text.append(text or "")
    return "\n\n".join(pages_text).strip()


def _extract_pdf_text_pdfplumber_ocr(pdf_path: Path) -> str:
    if pdfplumber is None:
        return ""
    pages_text: List[str] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            text = _ocr_page(page)
            pages_text.append(text or "")
    return "\n\n".join(pages_text).strip()


def _extract_pdf_text_pypdf2(pdf_path: Path) -> str:
    if PyPDF2 is None:
        return ""
    try:
        reader = PyPDF2.PdfReader(str(pdf_path))
    except Exception:
        return ""
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n\n".join(pages).strip()


def _extract_docx_text(docx_path: Path) -> str:
    if Document is None:
        raise RuntimeError("python-docx is not installed.")
    doc = Document(str(docx_path))
    lines: List[str] = []
    for para in doc.paragraphs:
        if not para.text:
            continue
        lines.append(para.text)
    return "\n".join(lines).strip()


def _extract_image_text(image_path: Path) -> str:
    if pytesseract is None or Image is None:
        raise RuntimeError("pytesseract or Pillow is not installed.")
    try:
        image = Image.open(str(image_path))
    except Exception as exc:
        raise RuntimeError(f"Failed to open image: {image_path}") from exc
    return pytesseract.image_to_string(image)


def _with_retries(fn, attempts: int = 3) -> str:
    last_exc = None
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            sleep_for = 2 ** attempt
            print(f"Extraction attempt {attempt + 1} failed: {exc}")
            if attempt < attempts - 1:
                time.sleep(sleep_for)
    raise RuntimeError(f"Extraction failed after {attempts} attempts") from last_exc


def extract_text_task(note_id: int) -> str:
    """
    Extract text from a note (PDF/DOCX/image/text).
    Updates the notes table with raw_text/cleaned_text/content_hash and status='extracted'.
    """
    note = get_note(note_id)
    if not note:
        raise ValueError(f"Note not found: {note_id}")

    file_path = _resolve_file_path(note)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    mime_type = note.get("mime_type") or _guess_mime_type(str(file_path))

    def _extract() -> str:
        if mime_type == "application/pdf" or str(file_path).lower().endswith(".pdf"):
            text = _extract_pdf_text_pdfplumber(file_path)
            if not text:
                text = _extract_pdf_text_pypdf2(file_path)
            if not text:
                text = _extract_pdf_text_pdfplumber_ocr(file_path)
            return text
        if "docx" in mime_type or str(file_path).lower().endswith(".docx"):
            return _extract_docx_text(file_path)
        if mime_type.startswith("image/") or mime_type == "image":
            return _extract_image_text(file_path)
        if mime_type.startswith("text/") or str(file_path).lower().endswith((".txt", ".md", ".csv", ".tsv")):
            return _read_text_file(file_path)
        return _read_text_file(file_path)

    raw_text = _with_retries(_extract, attempts=3).strip()
    cleaned_text = _clean_text(raw_text)
    content_hash = _hash_text(cleaned_text)

    update_note(
        note_id,
        raw_text=raw_text,
        cleaned_text=cleaned_text,
        content_hash=content_hash,
        mime_type=mime_type,
        status="extracted",
    )

    extracted_path, _ = _derived_paths(note)
    extracted_path.parent.mkdir(parents=True, exist_ok=True)
    extracted_path.write_text(cleaned_text, encoding="utf-8")

    return cleaned_text


def _token_offsets(text: str, encoding_name: str) -> List[int]:
    encoding = tiktoken.get_encoding(encoding_name)
    token_ids = encoding.encode(text)
    offsets = [0]
    for token_str in (encoding.decode([tok]) for tok in token_ids):
        offsets.append(offsets[-1] + len(token_str))
    return offsets


def _window_ranges(n_tokens: int, window_size: int, overlap: int) -> List[Tuple[int, int]]:
    if window_size <= 0:
        raise ValueError("window_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= window_size:
        raise ValueError("overlap must be < window_size")

    ranges: List[Tuple[int, int]] = []
    step = window_size - overlap
    start = 0
    while start < n_tokens:
        end = min(start + window_size, n_tokens)
        ranges.append((start, end))
        if end == n_tokens:
            break
        start += step
    return ranges


def chunk_text_task(
    note_id: int,
    window_size: int = 800,
    overlap: int = 80,
    min_tokens: int = 400,
    encoding_name: str = "cl100k_base",
) -> int:
    """
    Chunk cleaned note text into spans and write to spans table.
    Returns number of spans created and sets note status='chunked'.
    """
    note = get_note(note_id)
    if not note:
        raise ValueError(f"Note not found: {note_id}")

    extracted_path, chunks_path = _derived_paths(note)
    if extracted_path.exists():
        text = extracted_path.read_text(encoding="utf-8")
    else:
        text = note.get("cleaned_text") or note.get("raw_text") or ""
    text = str(text).strip()
    if not text:
        update_note(note_id, status="chunked")
        delete_spans(note_id)
        return 0

    if spacy is None:
        raise RuntimeError("spaCy is not installed.")
    nlp = spacy.load("en_core_web_sm", exclude=["ner", "tagger", "lemmatizer"])
    doc = nlp(text)

    encoding = tiktoken.get_encoding(encoding_name)

    sentences: List[Dict[str, Union[int, str]]] = []
    for sent in doc.sents:
        sent_text = sent.text
        token_count = len(encoding.encode(sent_text))
        if token_count == 0:
            continue
        sentences.append(
            {
                "text": sent_text,
                "start_char": sent.start_char,
                "end_char": sent.end_char,
                "token_count": token_count,
            }
        )

    if not sentences:
        update_note(note_id, status="chunked")
        delete_spans(note_id)
        return 0

    prefix_tokens = [0]
    for sent in sentences:
        prefix_tokens.append(prefix_tokens[-1] + int(sent["token_count"]))

    spans: List[Dict[str, Union[int, str]]] = []
    chunk_records: List[Dict[str, Union[str, int]]] = []

    start_idx = 0
    chunk_index = 0
    while start_idx < len(sentences):
        current_tokens = 0
        end_idx = start_idx
        while end_idx < len(sentences):
            next_tokens = current_tokens + int(sentences[end_idx]["token_count"])
            if next_tokens > window_size and current_tokens >= min_tokens:
                break
            current_tokens = next_tokens
            end_idx += 1

        if end_idx == start_idx:
            end_idx = min(start_idx + 1, len(sentences))
            current_tokens = int(sentences[start_idx]["token_count"])

        chunk_start_char = int(sentences[start_idx]["start_char"])
        chunk_end_char = int(sentences[end_idx - 1]["end_char"])
        chunk_text = text[chunk_start_char:chunk_end_char]

        spans.append(
            {
                "chunk_index": chunk_index,
                "start_char": chunk_start_char,
                "end_char": chunk_end_char,
                "token_count": current_tokens,
                "text": chunk_text,
            }
        )
        chunk_records.append(
            {
                "chunk_id": f"span_{chunk_index + 1}",
                "text": chunk_text,
                "start_char": chunk_start_char,
                "end_char": chunk_end_char,
                "token_count": current_tokens,
                "position": chunk_index,
            }
        )

        step_tokens = max(1, int(current_tokens * 0.9))
        target_token = prefix_tokens[start_idx] + step_tokens
        next_idx = start_idx + 1
        while next_idx < len(prefix_tokens) and prefix_tokens[next_idx] < target_token:
            next_idx += 1
        start_idx = min(next_idx, len(sentences))
        chunk_index += 1

    delete_spans(note_id)
    insert_spans(note_id, spans)
    update_note(note_id, status="chunked")

    chunks_path.parent.mkdir(parents=True, exist_ok=True)
    with chunks_path.open("w", encoding="utf-8") as handle:
        for record in chunk_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return len(spans)
