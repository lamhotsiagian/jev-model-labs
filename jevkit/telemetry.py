"""jevkit.telemetry -- decision logging, drift and audit sampling (Chapter 8).

Type safety constrains the output SHAPE; it does not guarantee the answer is
right. Production correctness comes from what you measure:

* one structured log line per answer (question ID+version, model version,
  probabilities, confidence, latency, tokens, policy outcome);
* drift in probability distributions (PSI) and escalation rate per question;
* a random audit sample of auto-acted decisions routed to humans;
* a golden set run before any model or question upgrade.
"""
from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


@dataclass
class DecisionLog:
    ts: float
    workflow: str
    question: str               # versioned id: name@vN#hash
    model_version: str
    kind: str
    probabilities: Mapping[str, float]
    confidence: float | None
    outcome: str | None
    latency_ms: float | None
    input_tokens: int
    request_id: str = ""
    audited: bool = False
    extra: dict = field(default_factory=dict)


class DecisionLogger:
    """Append-only JSONL. In production ship the same records to your log pipeline."""

    def __init__(self, path: str | Path, audit_rate: float = 0.02, seed: int | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.audit_rate = audit_rate
        self._rng = random.Random(seed)

    def log(self, rec: DecisionLog) -> DecisionLog:
        # Only auto-acted decisions are sampled: escalations are already reviewed.
        rec.audited = rec.outcome == "act" and self._rng.random() < self.audit_rate
        with self.path.open("a") as f:
            f.write(json.dumps(asdict(rec)) + "\n")
        return rec

    def read(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(l) for l in self.path.read_text().splitlines() if l.strip()]


def psi(expected: Sequence[float], actual: Sequence[float], bins: int = 10) -> float:
    """Population Stability Index between two samples of a probability signal.

    Rule of thumb: < 0.1 stable, 0.1-0.25 moderate shift, > 0.25 investigate.
    """
    edges = np.linspace(0, 1, bins + 1)
    e, _ = np.histogram(np.clip(expected, 0, 1), edges)
    a, _ = np.histogram(np.clip(actual, 0, 1), edges)
    e = np.maximum(e / max(1, e.sum()), 1e-4)
    a = np.maximum(a / max(1, a.sum()), 1e-4)
    return float(np.sum((a - e) * np.log(a / e)))


def drift_report(baseline: Iterable[dict], current: Iterable[dict]) -> list[dict]:
    """Per-question PSI on the headline signal plus escalation-rate change."""
    def signal(r: dict) -> float:
        if r["kind"] == "noul":
            return r["probabilities"]["true"]
        return r["confidence"]

    def group(rows):
        g: dict[str, list[dict]] = {}
        for r in rows:
            g.setdefault(r["question"].split("@")[0], []).append(r)
        return g

    b, c = group(baseline), group(current)
    out = []
    for q in sorted(set(b) & set(c)):
        esc_b = np.mean([r["outcome"] != "act" for r in b[q]])
        esc_c = np.mean([r["outcome"] != "act" for r in c[q]])
        p = psi([signal(r) for r in b[q]], [signal(r) for r in c[q]])
        out.append({"question": q, "psi": round(p, 3), "escalation_base": round(float(esc_b), 3),
                    "escalation_now": round(float(esc_c), 3),
                    "status": "ALERT" if p > 0.25 or abs(esc_c - esc_b) > 0.10 else
                              ("WATCH" if p > 0.10 else "OK")})
    return out


def golden_set_gate(results_old: Mapping[str, bool], results_new: Mapping[str, bool],
                    max_regression: float = 0.01, min_items: int = 50) -> dict:
    """Upgrade gate: the candidate version may not lose more than ``max_regression`` accuracy.

    Also counts flips (right -> wrong) because an unchanged average can hide
    a large number of swapped errors on specific segments.
    """
    ids = sorted(set(results_old) & set(results_new))
    if len(ids) < min_items:
        return {"pass": False, "reason": f"golden set too small ({len(ids)} < {min_items})"}
    acc_o = sum(results_old[i] for i in ids) / len(ids)
    acc_n = sum(results_new[i] for i in ids) / len(ids)
    flips_bad = sum(results_old[i] and not results_new[i] for i in ids)
    flips_good = sum(not results_old[i] and results_new[i] for i in ids)
    ok = acc_n >= acc_o - max_regression
    return {"pass": ok, "acc_old": round(acc_o, 4), "acc_new": round(acc_n, 4),
            "right_to_wrong": flips_bad, "wrong_to_right": flips_good, "n": len(ids),
            "reason": "within budget" if ok else "accuracy regression exceeds budget"}


def now() -> float:
    return time.time()
