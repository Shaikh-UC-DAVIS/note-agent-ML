# ml/task_schema.py

from __future__ import annotations
from typing import Optional, Literal, Dict, Any
from pydantic import BaseModel, Field


TaskStatus = Literal["todo", "in_progress", "done", "archived"]
TaskPriority = Literal["low", "medium", "high"]


class TaskRecord(BaseModel):
    id: str = Field(..., description="Unique task ID")
    title: str = Field(..., description="Human readable task title")
    canonical_text: Optional[str] = Field(default=None, description="Canonical text from extractor")

    note_id: Optional[str] = None
    user_id: Optional[str] = None
    workspace_id: Optional[str] = None

    due_date: Optional[str] = Field(default=None, description="YYYY-MM-DD")
    due_time: Optional[str] = Field(default=None, description="HH:MM")

    status: TaskStatus = "todo"
    priority: TaskPriority = "medium"

    confidence: Optional[float] = None
    source_text: Optional[str] = None

    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    completed_at: Optional[str] = None

    def to_storage_dict(self) -> Dict[str, Any]:
        return self.model_dump()
