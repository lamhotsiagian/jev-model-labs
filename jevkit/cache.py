"""jevkit.cache -- result cache keyed on hashed state + question version (Chapter 7).

A Jev answer is a pure function of (model version, state, question text).
Cache on exactly that tuple and nothing else:

* key includes the *concrete* model version, so an upgrade invalidates entries;
* key includes each question's content hash, so a wording change invalidates;
* key never includes the raw state, only its hash (no PII in the cache index).
"""
from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from typing import Any

from .serializers import to_state_json


def cache_key(model_version: str, state: Any, question_versions: list[str]) -> str:
    s = state if isinstance(state, str) else to_state_json(state)
    h = hashlib.sha256()
    h.update(model_version.encode())
    h.update(hashlib.sha256(s.encode()).digest())
    for qv in sorted(question_versions):
        h.update(qv.encode())
    return h.hexdigest()


class TTLCache:
    """Small in-process LRU with TTL. Swap for Redis in multi-replica deployments."""

    def __init__(self, max_items: int = 10_000, ttl_s: float = 3600):
        self.max_items, self.ttl_s = max_items, ttl_s
        self._d: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self.hits = self.misses = 0

    def get(self, key: str) -> Any | None:
        item = self._d.get(key)
        if item is None or time.monotonic() - item[0] > self.ttl_s:
            self.misses += 1
            self._d.pop(key, None)
            return None
        self._d.move_to_end(key)
        self.hits += 1
        return item[1]

    def put(self, key: str, value: Any) -> None:
        self._d[key] = (time.monotonic(), value)
        self._d.move_to_end(key)
        while len(self._d) > self.max_items:
            self._d.popitem(last=False)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0
