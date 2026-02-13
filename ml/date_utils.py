# ml/date_utils.py

from __future__ import annotations
from typing import Optional, Tuple
from datetime import datetime, date, timedelta
import re


def parse_natural_due_datetime(
    text: str,
    reference_datetime: Optional[datetime] = None
) -> Tuple[Optional[str], Optional[str], float]:
    """
    Parse natural language due date/time from text.

    Returns:
      (due_date_iso, due_time_24h, confidence)
      due_date_iso: YYYY-MM-DD or None
      due_time_24h: HH:MM or None
      confidence: float in [0,1]
    """
    ref_dt = reference_datetime or datetime.utcnow()
    t = (text or "").lower()

    due_date: Optional[date] = None
    due_time: Optional[str] = None
    confidence = 0.30

    # Relative dates
    if "today" in t:
        due_date = ref_dt.date()
        confidence = max(confidence, 0.85)
    elif "tomorrow" in t:
        due_date = (ref_dt + timedelta(days=1)).date()
        confidence = max(confidence, 0.90)
    elif "tonight" in t:
        due_date = ref_dt.date()
        due_time = "20:00"
        confidence = max(confidence, 0.80)

    # "next Monday"
    m = re.search(r"\bnext\s+(monday|mon|tuesday|tue|tues|wednesday|wed|thursday|thu|thur|thurs|friday|fri|saturday|sat|sunday|sun)\b", t)
    if m:
        wd = _weekday_to_int(m.group(1))
        due_date = _next_weekday(ref_dt.date(), wd)
        confidence = max(confidence, 0.88)

    # "Feb 20" / "February 20, 2026"
    m = re.search(
        r"\b(?:on|by|due\s+on|due\s+by)?\s*"
        r"(jan|january|feb|february|mar|march|apr|april|may|jun|june|jul|july|aug|august|sep|sept|september|oct|october|nov|november|dec|december)"
        r"\s+(\d{1,2})(?:st|nd|rd|th)?(?:,\s*(\d{4}))?\b",
        t
    )
    if m:
        month = _month_to_int(m.group(1))
        day = int(m.group(2))
        year = int(m.group(3)) if m.group(3) else ref_dt.year
        candidate = date(year, month, day)
        if m.group(3) is None and candidate < ref_dt.date():
            candidate = date(ref_dt.year + 1, month, day)
        due_date = candidate
        confidence = max(confidence, 0.92)

    # numeric date MM/DD or MM/DD/YYYY
    m = re.search(r"\b(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?\b", t)
    if m:
        mm, dd = int(m.group(1)), int(m.group(2))
        yy = m.group(3)
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
        confidence = max(confidence, 0.90)

    # Time: 3pm / 3:30 pm / 15:30
    tm = re.search(
        r"\b(at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b|\b(at\s+)?([01]?\d|2[0-3]):([0-5]\d)\b",
        t
    )
    if tm:
        if tm.group(2) is not None:
            h = int(tm.group(2))
            minute = int(tm.group(3)) if tm.group(3) else 0
            ampm = tm.group(4)
            if ampm == "pm" and h != 12:
                h += 12
            if ampm == "am" and h == 12:
                h = 0
            due_time = f"{h:02d}:{minute:02d}"
        else:
            h = int(tm.group(6))
            minute = int(tm.group(7))
            due_time = f"{h:02d}:{minute:02d}"

        confidence = max(confidence, 0.88 if due_date else 0.72)

    return (due_date.isoformat() if due_date else None, due_time, confidence)


def _weekday_to_int(day: str) -> int:
    d = day[:3].lower()
    mapping = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}
    return mapping[d]


def _next_weekday(d: date, target_weekday: int) -> date:
    days_ahead = (target_weekday - d.weekday()) % 7
    if days_ahead == 0:
        days_ahead = 7
    return d + timedelta(days=days_ahead)


def _month_to_int(month_str: str) -> int:
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
