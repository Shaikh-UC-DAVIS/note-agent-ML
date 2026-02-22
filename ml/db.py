"""
db.py

SQLite helpers for notes and spans used by extraction and chunking tasks.
"""

import os
import sqlite3
from typing import Any, Dict, Iterable, List, Optional


def _db_path() -> str:
    return os.environ.get("NOTE_AGENT_DB_PATH", "notes.db")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(_db_path())
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'uploaded',
                raw_text TEXT,
                cleaned_text TEXT,
                content_hash TEXT,
                workspace_id TEXT,
                file_id TEXT,
                mime_type TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS spans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                note_id INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                start_char INTEGER NOT NULL,
                end_char INTEGER NOT NULL,
                token_count INTEGER NOT NULL,
                text TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(note_id, chunk_index),
                FOREIGN KEY(note_id) REFERENCES notes(id)
            )
            """
        )

        _ensure_columns(
            conn,
            "notes",
            {
                "content_hash": "TEXT",
                "workspace_id": "TEXT",
                "file_id": "TEXT",
                "mime_type": "TEXT",
            },
        )


def _ensure_columns(conn: sqlite3.Connection, table: str, columns: Dict[str, str]) -> None:
    existing = {
        row["name"]
        for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
    }
    for name, col_type in columns.items():
        if name in existing:
            continue
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {col_type}")


def create_note(file_path: str, status: str = "uploaded") -> int:
    init_db()
    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO notes (file_path, status)
            VALUES (?, ?)
            """,
            (file_path, status),
        )
        conn.commit()
        return int(cur.lastrowid)


def get_note(note_id: int) -> Optional[Dict[str, Any]]:
    init_db()
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM notes WHERE id = ?",
            (note_id,),
        ).fetchone()
        return dict(row) if row else None


def update_note(note_id: int, **fields: Any) -> None:
    if not fields:
        return
    init_db()
    columns = ", ".join([f"{k} = ?" for k in fields.keys()] + ["updated_at = CURRENT_TIMESTAMP"])
    values: List[Any] = list(fields.values())
    values.append(note_id)
    with _connect() as conn:
        conn.execute(
            f"UPDATE notes SET {columns} WHERE id = ?",
            values,
        )
        conn.commit()


def delete_spans(note_id: int) -> None:
    init_db()
    with _connect() as conn:
        conn.execute("DELETE FROM spans WHERE note_id = ?", (note_id,))
        conn.commit()


def insert_spans(note_id: int, spans: Iterable[Dict[str, Any]]) -> None:
    init_db()
    rows = [
        (
            note_id,
            span["chunk_index"],
            span["start_char"],
            span["end_char"],
            span["token_count"],
            span["text"],
        )
        for span in spans
    ]
    if not rows:
        return
    with _connect() as conn:
        conn.executemany(
            """
            INSERT INTO spans (
                note_id, chunk_index, start_char, end_char, token_count, text
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
