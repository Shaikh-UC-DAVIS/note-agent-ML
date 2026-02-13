from typing import List, Optional, Literal, Dict, Any, Tuple
from pydantic import BaseModel, Field
from datetime import datetime, date

import re

from .date_utils import parse_natural_due_datetime


# -----------------------------
# Structured output models
# -----------------------------

class ExtractedObject(BaseModel):
    id: str = Field(description="Unique identifier for the object")
    type: Literal['Idea', 'Claim', 'Assumption', 'Question', 'Task', 'Evidence', 'Definition']
    canonical_text: str = Field(description="The concise, canonical text representation of the object")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0")
    # NEW: optional metadata for richer downstream use (calendar/task UI, ownership, etc.)
    attributes: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional structured metadata (e.g., due_date, due_time, note_id, user_id)"
    )


class Link(BaseModel):
    source_id: str
    target_id: str
    type: Literal['Supports', 'Contradicts', 'Refines', 'DependsOn', 'SameAs', 'Causes']
    confidence: float


class ExtractionResult(BaseModel):
    objects: List[ExtractedObject]
    links: List[Link]


class LLMExtractor:
    """
    NOTE:
    This remains a mock extractor (no live LLM call), but now includes robust task/date heuristics
    so your backend can persist and render calendar tasks.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4-turbo"):
        """
        Initialize the LLMExtractor.

        Args:
            api_key: OpenAI API key. If None, expects OPENAI_API_KEY env var.
            model: The LLM model to use.
        """
        self.model = model
        # In a real implementation, initialize OpenAI client here.
        # self.client = OpenAI(api_key=api_key)

    # -----------------------------
    # Public API
    # -----------------------------

    def extract(
        self,
        text: str,
        note_id: Optional[str] = None,
        user_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        reference_datetime: Optional[datetime] = None
    ) -> ExtractionResult:
        """
        Extracts structured objects and links from the given text.

        Args:
            text: Unstructured note text.
            note_id: Optional note identifier to stamp on extracted tasks.
            user_id: Optional user identifier for ownership correlation.
            workspace_id: Optional workspace identifier for multi-tenant scoping.
            reference_datetime: Optional reference datetime for parsing relative dates.

        Returns:
            ExtractionResult with objects and links.
        """
        ref_dt = reference_datetime or datetime.utcnow()
        lower_text = text.lower()

        objects: List[ExtractedObject] = []
        links: List[Link] = []

        # Keep existing mock behavior for compatibility with prior demos
        self._add_legacy_demo_entities(lower_text, objects, links)

        # New: task extraction pipeline
        task_objects, task_links = self._extract_tasks_from_text(
            text=text,
            note_id=note_id,
            user_id=user_id,
            workspace_id=workspace_id,
            reference_datetime=ref_dt
        )
        objects.extend(task_objects)
        links.extend(task_links)

        return ExtractionResult(objects=objects, links=links)

    def _construct_prompt(self, text: str) -> str:
        """Constructs the prompt for the LLM (kept for future live integration)."""
        return f"""
        Analyze the following text and extract structured knowledge objects.

        Text:
        {text}

        Output JSON matching the ExtractionResult schema.
        """

    # -----------------------------
    # Legacy demo entities (existing behavior)
    # -----------------------------

    def _add_legacy_demo_entities(
        self,
        lower_text: str,
        objects: List[ExtractedObject],
        links: List[Link]
    ) -> None:
        if "earth" in lower_text:
            objects.append(ExtractedObject(
                id="claim-earth-round",
                type="Claim",
                canonical_text="The earth is round",
                confidence=0.95
            ))
            objects.append(ExtractedObject(
                id="claim-earth-flat",
                type="Claim",
                canonical_text="The earth is flat",
                confidence=0.40
            ))
            links.append(Link(
                source_id="claim-earth-round",
                target_id="claim-earth-flat",
                type="Contradicts",
                confidence=0.90
            ))

        if "gravity" in lower_text:
            objects.append(ExtractedObject(
                id="idea-gravity",
                type="Idea",
                canonical_text="Gravity pulls everything towards the center of mass",
                confidence=0.90
            ))
            if "earth" in lower_text:
                links.append(Link(
                    source_id="idea-gravity",
                    target_id="claim-earth-round",
                    type="Supports",
                    confidence=0.85
                ))

    # -----------------------------
    # Task extraction internals
    # -----------------------------

    def _extract_tasks_from_text(
        self,
        text: str,
        note_id: Optional[str],
        user_id: Optional[str],
        workspace_id: Optional[str],
        reference_datetime: datetime
    ) -> Tuple[List[ExtractedObject], List[Link]]:
        objects: List[ExtractedObject] = []
        links: List[Link] = []

        # Split note into candidate segments (newline + sentence-ish splitting)
        segments = self._split_candidate_segments(text)

        # Verb cues for imperative task-like lines
        task_verbs = (
            "finish", "submit", "send", "complete", "review", "call", "email",
            "prepare", "schedule", "book", "attend", "pay", "buy", "update",
            "fix", "deploy", "ship", "write", "plan", "meet"
        )

        prev_task_id: Optional[str] = None
        task_counter = 1
        date_counter = 1

        for seg in segments:
            seg_clean = seg.strip(" -•\t")
            if not seg_clean:
                continue

            seg_lower = seg_clean.lower()

            # Task detection heuristic:
            # - has explicit task cue ("todo", "task", "by", "due")
            # - or starts with verb
            # - or checkbox style list
            looks_like_task = (
                "todo" in seg_lower
                or "task" in seg_lower
                or "due" in seg_lower
                or bool(re.match(r"^\[?\s?[x ]?\s?\]?\s*", seg_lower))
                or seg_lower.startswith(task_verbs)
                or any(seg_lower.startswith(v + " ") for v in task_verbs)
            )

            # Also accept if date cue appears in a likely action sentence
            has_date_cue = bool(
                re.search(
                    r"\b(today|tomorrow|tonight|next\s+(mon|tue|wed|thu|fri|sat|sun)\w*|"
                    r"on\s+[A-Za-z]{3,9}\s+\d{1,2}|by\s+[A-Za-z]{3,9}\s+\d{1,2}|"
                    r"\d{1,2}/\d{1,2}(/\d{2,4})?)\b",
                    seg_lower
                )
            )

            if not (looks_like_task or has_date_cue):
                continue

            due_date_iso, due_time_24h, date_conf = parse_natural_due_datetime(
                seg_clean,
                reference_datetime
        )


            # Remove obvious time/date phrasing from title for cleaner canonical text
            canonical_title = self._clean_task_title(seg_clean)

            # Confidence composition
            conf = 0.55
            if looks_like_task:
                conf += 0.20
            if due_date_iso:
                conf += 0.15
            if due_time_24h:
                conf += 0.05
            conf = min(conf, 0.99)

            task_id = self._slug_id("task", note_id, task_counter, canonical_title)
            task_counter += 1

            task_attrs: Dict[str, Any] = {
                "status": "todo",
                "priority": "medium",
                "source_text": seg_clean,
                "note_id": note_id,
                "user_id": user_id,
                "workspace_id": workspace_id,
                "due_date": due_date_iso,
                "due_time": due_time_24h,
                "date_parse_confidence": date_conf
            }

            objects.append(ExtractedObject(
                id=task_id,
                type="Task",
                canonical_text=canonical_title,
                confidence=conf,
                attributes=task_attrs
            ))

            # Optional: represent due date as a Definition node for graph compatibility
            if due_date_iso:
                date_obj_id = self._slug_id("date", note_id, date_counter, due_date_iso)
                date_counter += 1

                objects.append(ExtractedObject(
                    id=date_obj_id,
                    type="Definition",
                    canonical_text=due_date_iso,
                    confidence=max(0.60, date_conf),
                    attributes={
                        "kind": "date",
                        "date": due_date_iso,
                        "time": due_time_24h
                    }
                ))

                links.append(Link(
                    source_id=task_id,
                    target_id=date_obj_id,
                    type="Refines",   # "Task is refined by due-date definition"
                    confidence=max(0.65, date_conf)
                ))

            # Optional chaining: neighboring task list items as DependsOn
            if prev_task_id:
                links.append(Link(
                    source_id=prev_task_id,
                    target_id=task_id,
                    type="DependsOn",
                    confidence=0.35
                ))

            prev_task_id = task_id

        return objects, links

    # -----------------------------
    # Date/time parsing helpers
    # -----------------------------

    def _parse_due_datetime(self, text: str, ref_dt: datetime) -> Tuple[Optional[str], Optional[str], float]:
        """
        Returns:
            due_date_iso: YYYY-MM-DD or None
            due_time_24h: HH:MM or None
            confidence: float
        """
        t = text.lower()
        conf = 0.35
        due_date: Optional[date] = None
        due_time: Optional[str] = None

        # Relative day keywords
        if "today" in t:
            due_date = ref_dt.date()
            conf = 0.85
        elif "tomorrow" in t:
            due_date = (ref_dt + timedelta(days=1)).date()
            conf = 0.90
        elif "tonight" in t:
            due_date = ref_dt.date()
            due_time = "20:00"
            conf = 0.80

        # next weekday
        weekday_match = re.search(r"\bnext\s+(monday|mon|tuesday|tue|tues|wednesday|wed|thursday|thu|thur|thurs|friday|fri|saturday|sat|sunday|sun)\b", t)
        if weekday_match:
            target_weekday = self._weekday_to_int(weekday_match.group(1))
            due_date = self._next_weekday(ref_dt.date(), target_weekday)
            conf = max(conf, 0.88)

        # explicit month day (e.g., Feb 20, February 20)
        month_day_match = re.search(
            r"\b(?:on|by|due\s+on|due\s+by)?\s*"
            r"(jan|january|feb|february|mar|march|apr|april|may|jun|june|jul|july|aug|august|sep|sept|september|oct|october|nov|november|dec|december)"
            r"\s+(\d{1,2})(?:st|nd|rd|th)?(?:,\s*(\d{4}))?\b",
            t
        )
        if month_day_match:
            month_str = month_day_match.group(1)
            day_num = int(month_day_match.group(2))
            year_str = month_day_match.group(3)

            month_num = self._month_to_int(month_str)
            year_num = int(year_str) if year_str else ref_dt.year

            # If no year and date already passed this year, roll to next year
            candidate = date(year_num, month_num, day_num)
            if year_str is None and candidate < ref_dt.date():
                candidate = date(ref_dt.year + 1, month_num, day_num)

            due_date = candidate
            conf = max(conf, 0.92)

        # numeric date (MM/DD or MM/DD/YYYY)
        numeric_match = re.search(r"\b(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?\b", t)
        if numeric_match:
            mm = int(numeric_match.group(1))
            dd = int(numeric_match.group(2))
            yy = numeric_match.group(3)

            if yy is None:
                year = ref_dt.year
                candidate = date(year, mm, dd)
                if candidate < ref_dt.date():
                    candidate = date(year + 1, mm, dd)
            else:
                year = int(yy)
                if year < 100:
                    year += 2000
                candidate = date(year, mm, dd)

            due_date = candidate
            conf = max(conf, 0.90)

        # time parsing (3pm, 3:30 pm, 15:30)
        time_match = re.search(
            r"\b(at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b|\b(at\s+)?([01]?\d|2[0-3]):([0-5]\d)\b",
            t
        )
        if time_match:
            # 12-hour format branch
            if time_match.group(2) is not None:
                hour = int(time_match.group(2))
                minute = int(time_match.group(3)) if time_match.group(3) else 0
                ampm = time_match.group(4)
                if ampm == "pm" and hour != 12:
                    hour += 12
                if ampm == "am" and hour == 12:
                    hour = 0
                due_time = f"{hour:02d}:{minute:02d}"
            else:
                # 24-hour branch
                hour = int(time_match.group(6))
                minute = int(time_match.group(7))
                due_time = f"{hour:02d}:{minute:02d}"

            conf = max(conf, 0.88 if due_date else 0.72)

        due_date_iso = due_date.isoformat() if due_date else None
        return due_date_iso, due_time, conf

    # -----------------------------
    # Generic helpers
    # -----------------------------

    def _split_candidate_segments(self, text: str) -> List[str]:
        # Keep list-style lines + sentence fragments
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        segments: List[str] = []

        for ln in lines:
            # If likely bullet/checklist, keep intact
            if re.match(r"^(\-|\*|•|\d+\.)\s+", ln) or re.match(r"^\[\s?[xX ]\s?\]\s*", ln):
                segments.append(ln)
            else:
                # split by sentence punctuation while preserving meaningful fragments
                parts = re.split(r"(?<=[.!?])\s+", ln)
                segments.extend([p.strip() for p in parts if p.strip()])

        return segments

    def _infer_priority(self, lower_text: str) -> str:
        if any(k in lower_text for k in ("urgent", "asap", "immediately", "critical", "high priority")):
            return "high"
        if any(k in lower_text for k in ("low priority", "whenever", "later")):
            return "low"
        return "medium"

    def _clean_task_title(self, text: str) -> str:
        t = text.strip()

        # Remove checkbox/bullet prefixes
        t = re.sub(r"^(\-|\*|•|\d+\.)\s*", "", t)
        t = re.sub(r"^\[\s?[xX ]\s?\]\s*", "", t)

        # Remove obvious date/time phrases
        t = re.sub(r"\b(by|due|on)\b\s+[^,.;]+", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\bat\s+\d{1,2}(:\d{2})?\s*(am|pm)?\b", "", t, flags=re.IGNORECASE)

        # Collapse spaces and trailing punctuation
        t = re.sub(r"\s+", " ", t).strip(" .,-")
        return t if t else text.strip()

    def _slug_id(self, prefix: str, note_id: Optional[str], idx: int, text: str) -> str:
        core = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
        core = core[:40] if core else "item"
        nid = note_id or "note"
        return f"{prefix}-{nid}-{idx}-{core}"

    def _weekday_to_int(self, day: str) -> int:
        d = day[:3].lower()
        mapping = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}
        return mapping[d]

    def _next_weekday(self, d: date, target_weekday: int) -> date:
        days_ahead = (target_weekday - d.weekday()) % 7
        if days_ahead == 0:
            days_ahead = 7
        return d + timedelta(days=days_ahead)

    def _month_to_int(self, month_str: str) -> int:
        m = month_str.strip().lower()
        month_map = {
            "jan": 1, "january": 1,
            "feb": 2, "february": 2,
            "mar": 3, "march": 3,
            "apr": 4, "april": 4,
            "may": 5,
            "jun": 6, "june": 6,
            "jul": 7, "july": 7,
            "aug": 8, "august": 8,
            "sep": 9, "sept": 9, "september": 9,
            "oct": 10, "october": 10,
            "nov": 11, "november": 11,
            "dec": 12, "december": 12
        }
        return month_map[m]



## Changes Made
'''

attributes added to ExtractedObject
So a Task can carry due-date/user/note/workspace metadata without changing the object type enum.

extract(...) signature expanded
Optional note_id, user_id, workspace_id, reference_datetime let your ingestion layer pass ownership/date context.

Task extraction heuristics
Parses imperative or checklist lines and date cues.

Date/time normalization
Converts to:

due_date: YYYY-MM-DD

due_time: HH:MM

Graph-friendly links
For each task with a date, creates a Definition date object and Refines link.

'''