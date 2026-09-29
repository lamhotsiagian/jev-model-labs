"""jevkit.client -- one client, three interchangeable backends (Chapters 1, 2, 7).

    sim     offline JevSimulator (default; no key, deterministic, CI-safe)
    http    raw REST: POST https://api.typesafe.ai/v1/systemone
    replay  recorded fixtures on disk (unit tests never hit the API)

Choose with ``JEV_BACKEND=sim|http|replay``; the HTTP backend reads
``TYPESAFE_API_KEY``. Every call returns a typed ``JevResult`` and records the
concrete model version and token usage.

Retry policy follows the documented guidance: back off exponentially on 429
(rate limited) and 529 (overloaded); never retry 401 or 422, which are
caller bugs that a retry cannot fix.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Callable, Mapping

from .contract import JevResult, Question, build_request, parse_response
from .simulator import JevSimulator

API_URL = os.environ.get("TYPESAFE_BASE_URL", "https://api.typesafe.ai/v1") + "/systemone"
RETRYABLE = {429, 500, 502, 503, 529}


class JevError(RuntimeError):
    def __init__(self, status: int, message: str):
        super().__init__(f"HTTP {status}: {message}")
        self.status = status


def request_key(body: Mapping[str, Any]) -> str:
    """Stable hash of a request body. Used for fixtures and for the result cache."""
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:24]


# ---------------------------------------------------------------------------
# Backends: callables body -> raw response dict
# ---------------------------------------------------------------------------
class HttpBackend:
    def __init__(self, api_key: str | None = None, timeout_s: float = 5.0,
                 max_retries: int = 3, base_delay_s: float = 0.25):
        import httpx  # imported lazily so the sim path has no network dependency
        key = api_key or os.environ.get("TYPESAFE_API_KEY")
        if not key:
            raise RuntimeError("TYPESAFE_API_KEY is not set")
        self._client = httpx.Client(timeout=timeout_s,
                                    headers={"Authorization": f"Bearer {key}"})
        self.max_retries, self.base_delay_s = max_retries, base_delay_s

    def __call__(self, body: Mapping[str, Any]) -> dict:
        for attempt in range(self.max_retries + 1):
            r = self._client.post(API_URL, json=body)
            if r.status_code == 200:
                return r.json()
            if r.status_code not in RETRYABLE or attempt == self.max_retries:
                raise JevError(r.status_code, r.text[:300])
            # exponential backoff with full jitter
            time.sleep(random.uniform(0, self.base_delay_s * 2 ** attempt))
        raise AssertionError("unreachable")


class ReplayBackend:
    """Serve recorded responses. Missing fixture == test bug, so fail loudly."""

    def __init__(self, fixture_dir: str | Path):
        self.dir = Path(fixture_dir)

    def __call__(self, body: Mapping[str, Any]) -> dict:
        path = self.dir / f"{request_key(body)}.json"
        if not path.exists():
            raise FileNotFoundError(f"no fixture for request {path.name}; record it first")
        return json.loads(path.read_text())


class RecordingBackend:
    """Wrap any backend and write every response to disk as a replayable fixture."""

    def __init__(self, inner: Callable[[Mapping[str, Any]], dict], fixture_dir: str | Path):
        self.inner, self.dir = inner, Path(fixture_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def __call__(self, body: Mapping[str, Any]) -> dict:
        raw = self.inner(body)
        (self.dir / f"{request_key(body)}.json").write_text(json.dumps(raw, indent=2))
        return raw


def default_backend() -> Callable[[Mapping[str, Any]], dict]:
    kind = os.environ.get("JEV_BACKEND", "sim")
    if kind == "http":
        return HttpBackend()
    if kind == "replay":
        return ReplayBackend(os.environ.get("JEV_FIXTURES", "data/fixtures"))
    return JevSimulator(simulate_latency=os.environ.get("JEV_SIM_SLEEP") == "1")


# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------
class JevClient:
    def __init__(self, backend: Callable[[Mapping[str, Any]], dict] | None = None,
                 model: str = "jev-latest"):
        self.backend = backend or default_backend()
        self.model = model

    @property
    def is_simulated(self) -> bool:
        inner = getattr(self.backend, "inner", self.backend)
        return isinstance(inner, JevSimulator)

    def ask(self, state: Any, questions: Mapping[str, Question]) -> JevResult:
        body = build_request(state, questions, self.model)
        t0 = time.perf_counter()
        raw = self.backend(body)
        wall_ms = (time.perf_counter() - t0) * 1000
        # A simulator that does not sleep reports its modelled latency instead.
        inner = getattr(self.backend, "inner", self.backend)
        modelled = self.is_simulated and not inner.simulate_latency
        latency = raw.get("_sim_latency_ms", wall_ms) if modelled else wall_ms
        return parse_response(raw, questions, latency_ms=latency)


class AsyncJevClient:
    """Async twin used by the decision microservice (Chapter 7)."""

    def __init__(self, backend: Callable[[Mapping[str, Any]], dict] | None = None,
                 model: str = "jev-latest", timeout_s: float = 2.0):
        self._sync = JevClient(backend, model)
        self.timeout_s = timeout_s

    async def ask(self, state: Any, questions: Mapping[str, Question]) -> JevResult:
        return await asyncio.wait_for(asyncio.to_thread(self._sync.ask, state, questions),
                                      timeout=self.timeout_s)
