"""jevkit.serializers -- turn source data into compact, labelled state (Chapter 4).

Jev quality is determined mostly by two inputs you control: the questions and
the state. Good state is compact (only what the questions need), labelled
(every value has a name), and stable (the same record always serializes to
the same bytes, so caches and evaluations stay comparable).

Three serializers cover most production sources:
    serialize_row       one database / API record
    serialize_events    a time-ordered window of events
    serialize_chunk     one chunk of a long document with its location
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any, Iterable, Mapping, Sequence

_WS = re.compile(r"\s+")


def _clean(v: Any, max_chars: int) -> Any:
    if isinstance(v, str):
        v = _WS.sub(" ", v).strip()
        return v if len(v) <= max_chars else v[: max_chars - 1] + "…"
    if isinstance(v, datetime):
        return v.isoformat(timespec="minutes")
    return v


def serialize_row(row: Mapping[str, Any], fields: Sequence[str] | None = None,
                  rename: Mapping[str, str] | None = None, max_chars: int = 600,
                  as_json: bool = False) -> str | dict:
    """Project a record onto the fields the questions need, with readable labels.

    ``fields`` is an allow-list. Anything not listed never reaches the model:
    that is both a quality control (less distraction) and a privacy control
    (no accidental PII in state).
    """
    rename = rename or {}
    keep = fields or list(row)
    data = {rename.get(k, k): _clean(row[k], max_chars) for k in keep
            if k in row and row[k] not in (None, "", [], {})}
    if as_json:
        return data
    return "\n".join(f"{k}: {v}" for k, v in data.items())


def serialize_events(events: Iterable[Mapping[str, Any]], window: int = 20,
                     ts_key: str = "ts", type_key: str = "type",
                     detail_keys: Sequence[str] = ("detail",), now: datetime | None = None) -> str:
    """Render the most recent ``window`` events as one line each, newest last.

    Relative ages ("-12m") are computed in code because date arithmetic is a
    documented Jev weak spot: never ask the model to subtract timestamps.
    """
    evs = sorted(events, key=lambda e: e[ts_key])[-window:]
    now = now or (evs[-1][ts_key] if evs else datetime.now())
    lines = []
    for e in evs:
        age_min = int((now - e[ts_key]).total_seconds() // 60)
        detail = " ".join(str(e[k]) for k in detail_keys if e.get(k))
        lines.append(f"[-{age_min}m] {e[type_key]}: {_clean(detail, 200)}")
    return "events (oldest first):\n" + "\n".join(lines)


def serialize_chunk(text: str, doc_id: str, section: str, chunk_idx: int, n_chunks: int,
                    max_chars: int = 3000) -> str:
    """Wrap a document chunk with the location metadata a question may need."""
    return (f"document: {doc_id}\nsection: {section}\n"
            f"chunk: {chunk_idx + 1} of {n_chunks}\n---\n{_clean(text, max_chars)}")


def serialize_prose(row: Mapping[str, Any]) -> str:
    """The anti-pattern, kept for the A/B lab: an unlabelled prose blob."""
    return ". ".join(str(v) for v in row.values() if v not in (None, "")) + "."


def to_state_json(obj: Any) -> str:
    """Canonical JSON (sorted keys, no whitespace) so hashes are stable."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
