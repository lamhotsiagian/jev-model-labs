"""Chapter 7 lab -- the decision microservice.

    POST /decide/{workflow}          one record -> answers + per-action decisions
    POST /decide/{workflow}/batch    many small records, bounded concurrency
    GET  /healthz                    liveness + backend + model version seen
    GET  /metrics                    cache hit rate, calls, escalation counters

Design points (each maps to a section of Chapter 7):
  * questions come from the registry, thresholds from the policy file;
  * one fan-out call per record; async with a hard timeout;
  * retries with exponential backoff on 429/529/timeouts, never on 4xx bugs;
  * cache keyed on (model version, hashed state, question versions);
  * FAIL CLOSED: on outage or when JEV_FALLBACK=escalate every action escalates;
  * response carries full probabilities so callers can audit or re-threshold.

Run:  uvicorn labs.ch07_service.app:app --port 8080
Docs: http://localhost:8080/docs (OpenAPI)
"""
from __future__ import annotations

import asyncio
import os
import random
import time
import uuid
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from jevkit.cache import TTLCache, cache_key
from jevkit.client import AsyncJevClient, JevError
from jevkit.contract import ChoiceAnswer, NoulAnswer, ScoreAnswer
from jevkit.policy import Outcome, Policy
from jevkit.registry import QuestionRegistry
from jevkit.serializers import serialize_row
from jevkit.telemetry import DecisionLog, DecisionLogger, now

ROOT = Path(os.environ.get("JEV_LABS_ROOT", Path(__file__).resolve().parents[2]))
WF_DIR = Path(__file__).parent / "workflows"
MAX_RETRIES, TIMEOUT_S, BATCH_CONCURRENCY = 3, 2.0, 16


class Workflow:
    def __init__(self, path: Path):
        cfg = yaml.safe_load(path.read_text())
        self.name = cfg["workflow"]
        self.registry = QuestionRegistry.from_yaml(ROOT / cfg["registry"])
        self.policy = Policy.from_yaml(ROOT / cfg["policy"])
        self.fields = cfg.get("state_fields")
        self.bindings: dict[str, str] = cfg.get("bindings", {})
        self.questions = self.registry.questions()
        self.qversions = [q.versioned_id for q in self.registry]


class DecideRequest(BaseModel):
    record: dict[str, Any] = Field(..., examples=[{"plan": "pro", "customer_message":
                                                   "Checkout is down for all our customers!"}])
    request_id: str | None = None


class BatchRequest(BaseModel):
    records: list[dict[str, Any]]


app = FastAPI(title="Jev Decision Service", version="1.0.0")
WORKFLOWS = {wf.name: wf for wf in (Workflow(p) for p in sorted(WF_DIR.glob("*.yaml")))}
CLIENT = AsyncJevClient(timeout_s=TIMEOUT_S)
CACHE = TTLCache(max_items=50_000, ttl_s=3600)
LOGGER = DecisionLogger(os.environ.get("JEV_DECISION_LOG", "var/decisions.jsonl"), audit_rate=0.02)
STATS = {"calls": 0, "retries": 0, "fallbacks": 0, "last_model_version": None}


def answer_json(a) -> dict:
    if isinstance(a, NoulAnswer):
        return {"type": "noul", "p_true": a.p_true}
    if isinstance(a, ChoiceAnswer):
        return {"type": "choice", "choice": a.choice, "probabilities": a.probabilities,
                "confidence": a.confidence}
    if isinstance(a, ScoreAnswer):
        return {"type": "score", "score": a.score, "normalized": a.normalized,
                "probabilities": a.probabilities, "confidence": a.confidence}
    raise TypeError(type(a))


async def ask_with_retry(state: Any, wf: Workflow):
    for attempt in range(MAX_RETRIES + 1):
        try:
            STATS["calls"] += 1
            return await CLIENT.ask(state, wf.questions)
        except (asyncio.TimeoutError, JevError) as exc:
            retryable = isinstance(exc, asyncio.TimeoutError) or exc.status in (429, 529, 503)
            if not retryable or attempt == MAX_RETRIES:
                raise
            STATS["retries"] += 1
            await asyncio.sleep(random.uniform(0, 0.2 * 2 ** attempt))


def fallback(wf: Workflow, reason: str) -> dict:
    STATS["fallbacks"] += 1
    return {"workflow": wf.name, "fallback": True, "reason": reason,
            "decisions": {a: {"outcome": Outcome.ESCALATE.value, "reason": reason}
                          for a in wf.bindings.values()}}


async def decide_one(wf: Workflow, record: dict, request_id: str) -> dict:
    if os.environ.get("JEV_FALLBACK") == "escalate":            # kill switch / feature flag
        return fallback(wf, "fallback flag enabled")
    state = serialize_row(record, fields=wf.fields) if wf.fields else record
    key = cache_key(STATS["last_model_version"] or "unknown", state, wf.qversions)
    cached = CACHE.get(key)
    if cached is not None:
        return {**cached, "cached": True, "request_id": request_id}
    t0 = time.perf_counter()
    try:
        res = await ask_with_retry(state, wf)
    except Exception as exc:                                   # fail closed, never fail open
        return fallback(wf, f"jev unavailable: {type(exc).__name__}")
    STATS["last_model_version"] = res.model_version
    decisions = {}
    for qid, action in wf.bindings.items():
        d = wf.policy.decide(action, res[qid])
        decisions[action] = {"outcome": d.outcome.value, "signal": round(d.signal_value, 3),
                             "threshold": d.threshold, "question": qid}
        a = res[qid]
        LOGGER.log(DecisionLog(
            ts=now(), workflow=wf.name, question=wf.registry[qid].versioned_id,
            model_version=res.model_version, kind=answer_json(a)["type"],
            probabilities=({"true": a.p_true} if isinstance(a, NoulAnswer) else dict(a.probabilities)),
            confidence=None if isinstance(a, NoulAnswer) else a.confidence,
            outcome=d.outcome.value, latency_ms=res.latency_ms, input_tokens=res.input_tokens,
            request_id=request_id, extra={"action": action, "policy_version": wf.policy.version}))
    body = {"workflow": wf.name, "model_version": res.model_version, "policy_version": wf.policy.version,
            "answers": {q: answer_json(a) for q, a in res.answers.items()}, "decisions": decisions,
            "usage": {"input_tokens": res.input_tokens},
            "service_ms": round((time.perf_counter() - t0) * 1000, 1), "fallback": False}
    CACHE.put(cache_key(res.model_version, state, wf.qversions), body)
    return {**body, "cached": False, "request_id": request_id}


def get_wf(workflow: str) -> Workflow:
    if workflow not in WORKFLOWS:
        raise HTTPException(404, f"unknown workflow '{workflow}'")
    return WORKFLOWS[workflow]


@app.post("/decide/{workflow}")
async def decide(workflow: str, req: DecideRequest) -> dict:
    return await decide_one(get_wf(workflow), req.record, req.request_id or uuid.uuid4().hex)


@app.post("/decide/{workflow}/batch")
async def decide_batch(workflow: str, req: BatchRequest) -> dict:
    wf, sem = get_wf(workflow), asyncio.Semaphore(BATCH_CONCURRENCY)

    async def one(r):
        async with sem:
            return await decide_one(wf, r, uuid.uuid4().hex)
    results = await asyncio.gather(*(one(r) for r in req.records))
    return {"n": len(results), "results": results}


@app.get("/healthz")
async def healthz() -> dict:
    return {"ok": True, "workflows": list(WORKFLOWS), "model_version": STATS["last_model_version"],
            "backend": os.environ.get("JEV_BACKEND", "sim")}


@app.get("/metrics")
async def metrics() -> dict:
    return {**STATS, "cache_hit_rate": round(CACHE.hit_rate, 3)}
