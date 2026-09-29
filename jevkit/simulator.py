"""jevkit.simulator -- an offline, deterministic stand-in for the Jev API.

WHY THIS EXISTS. Jev is a hosted, early-access, single-vendor model. Labs,
unit tests and CI must not depend on a network call, an API key or a waitlist.
The simulator returns bodies with the *exact* wire shape of
``POST /v1/systemone`` so every jevkit module is exercised end to end.

WHAT IT IS NOT. It is not Jev. Its "judgment" is a transparent keyword
heuristic with seeded noise. It deliberately reproduces the *contract* and
several documented *behaviours* so the labs have something real to measure:

* all questions are answered in one call, each isolated from the others
  (the answer to question A depends only on state + A);
* latency is almost flat in question count and grows with state length;
* probabilities are rounded to 2 decimals and may not sum to 1;
* confidence is normalized peakedness of the returned distribution;
* the response reports a concrete version, not the ``jev-latest`` alias;
* it is steerable by injected instructions inside the state (a documented
  Jev 1.13 weak spot), which Chapter 9 defends against.

Every number a lab prints under this backend is labelled SIMULATED. Re-run
with ``JEV_BACKEND=http`` and a real key to measure Jev itself.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import re
import time
from typing import Any, Mapping

SIM_VERSION = "jev-1.13.0-sim"

_STOP = set("""a an the is are was were be been of to in on for and or not no yes this that
these those it its with as by at from does do did has have had any all which what who whom how
text state customer message ticket about there their them they we you your our than then
should would could can will into out over under more most less very""".split())
_WORD = re.compile(r"[a-z0-9]+")
_VAGUE = {"urgent", "important", "serious", "bad", "good", "priority", "problematic", "appropriate", "risky"}
_INJECT = re.compile(
    r"(?i)(ignore|disregard|override)[^.\n]{0,80}?\b(answer|choose|select|classify as|output)\s+['\"]?([a-z_0-9]+)")


def approx_tokens(obj: Any) -> int:
    """~4 characters per token. Good enough for budgeting and latency models."""
    text = obj if isinstance(obj, str) else json.dumps(obj, sort_keys=True)
    return max(1, len(text) // 4)


def _words(obj: Any) -> list[str]:
    text = obj if isinstance(obj, str) else json.dumps(obj, sort_keys=True)
    out = []
    for w in _WORD.findall(text.lower()):
        if w in _STOP or len(w) < 3:
            continue
        out.append(w[:6])            # crude stem: "frustrated"/"frustrating" -> "frustr"
    return out


def _rng(*parts: Any) -> random.Random:
    h = hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()
    return random.Random(int(h[:16], 16))


def _softmax(logits: list[float]) -> list[float]:
    m = max(logits)
    ex = [math.exp(x - m) for x in logits]
    s = sum(ex)
    return [e / s for e in ex]


def _overlap(keys: list[str], state_words: set[str]) -> float:
    if not keys:
        return 0.0
    hits = sum(1 for k in set(keys) if k in state_words)
    return hits / math.sqrt(len(set(keys)))


_FOCUS = re.compile(r"(?i)\b(item|candidate)\s+([a-z0-9_-]+)")


def _focus(q: Mapping[str, Any], state_text: str, state_words: set[str]) -> set[str]:
    """When a question names one element ("item 3", "candidate kb2") and the state
    lists elements as "item 3: ..." lines, the judgment attends to that line plus
    the shared context lines. This is how per-item and pairwise helpers stay
    independent inside one batched call."""
    m = _FOCUS.search(json.dumps(q.get("instructions", "")))
    if not m:
        return state_words
    kind, ident = m.group(1).lower(), m.group(2).lower()
    lines = state_text.splitlines()
    target = [l for l in lines if l.lower().startswith(f"{kind} {ident}:")]
    if not target:
        return state_words
    shared = [l for l in lines if not re.match(r"(?i)(item|candidate)\s+\S+:", l)]
    body = target[0].split(":", 1)[1]
    ctx = " ".join(l.split(":", 1)[-1] for l in shared)
    return set(_words(body + " " + ctx))


def _peakedness(ps: list[float]) -> float:
    k = len(ps)
    return max(0.0, min(1.0, (k * max(ps) - 1) / (k - 1)))


class JevSimulator:
    """Callable that maps a request body to a response body (dicts, wire format)."""

    def __init__(self, sharpness: float = 4.2, noise: float = 0.35,
                 simulate_latency: bool = False, version: str = SIM_VERSION):
        self.sharpness, self.noise = sharpness, noise
        self.simulate_latency = simulate_latency
        self.version = version

    # -- latency model: base + per-state-token + tiny per-question term --------
    def latency_ms(self, state_tokens: int, n_questions: int, rng: random.Random) -> float:
        return 62.0 + 0.028 * state_tokens + 0.35 * n_questions + rng.uniform(-6, 6)

    def __call__(self, body: Mapping[str, Any]) -> dict:
        state = body["state"]
        questions = body["questions"]
        state_text = state if isinstance(state, str) else json.dumps(state, sort_keys=True)
        # Field labels ("draft:", "ticket_id": ...) are structure, not evidence.
        labels = set(_words(" ".join(re.findall(r"(?m)^\s*([A-Za-z_][\w ]{0,30}):", state_text)
                                     + re.findall(r'"([A-Za-z_]\w*)"\s*:', state_text))))
        state_words = set(_words(state_text)) - labels
        injected = [(m.group(3).lower()) for m in _INJECT.finditer(state_text)]

        answers = {}
        for qid, q in questions.items():
            # Isolation: randomness depends on THIS question and the state only.
            # The main noise is seeded by the evidence the question can anchor on
            # (question vocabulary present in the state), so irrelevant edits do
            # not move it. A small jitter seeded by the full text models
            # sensitivity to surface form. It is large for vague, holistic wording
            # ("urgent", "important", ...), a deliberate modelling choice that
            # mirrors how an under-specified question leaves the judgment to
            # incidental features of the text.
            sw = _focus(q, state_text, state_words)
            anchor = sorted(set(_words(q)) & sw)
            rng = _rng(anchor, q)
            jit = _rng(state_text, q)
            vague = bool(_VAGUE & set(_WORD.findall(json.dumps(q.get("instructions", "")).lower())))
            sigma = 1.10 if vague else 0.05
            answers[qid] = self._answer(q, sw, injected, rng,
                                        lambda: jit.gauss(0, sigma))

        s_tok = approx_tokens(state_text)
        q_tok = sum(approx_tokens(q) for q in questions.values())
        out_tok = sum(1 + len(q.get("criteria") or []) for q in questions.values())
        lat = self.latency_ms(s_tok, len(questions), _rng(state_text, len(questions)))
        if self.simulate_latency:
            time.sleep(lat / 1000.0)
        return {"model": self.version, "answers": answers,
                "usage": {"input_tokens": s_tok + q_tok, "output_tokens": out_tok},
                "_sim_latency_ms": round(lat, 1)}

    # -------------------------------------------------------------------------
    def _answer(self, q: Mapping[str, Any], sw: set[str], injected: list[str],
                rng: random.Random, jitter) -> dict:
        kind = q["type"]
        instr = _words(q.get("instructions", ""))
        if kind == "noul":
            crit = q.get("criteria") or {}
            pos = instr + _words(crit.get("true", ""))
            neg = [w for w in _words(crit.get("false", "")) if w not in set(pos)]
            logit = 1.4 * self.sharpness * (_overlap(pos, sw) - 0.55 * _overlap(neg, sw)) - 1.3
            logit += rng.gauss(0, self.noise) + jitter()
            if any(t in ("yes", "true") for t in injected):
                logit += 3.5
            p = 1 / (1 + math.exp(-logit))
            return {"type": "noul", "noul": round(p, 2)}

        if kind == "choice":
            names = list(q["criteria"].keys())
            logits = []
            for name in names:
                desc = q["criteria"][name]
                keys = _words(name.replace("_", " ")) + _words(desc or "")
                base = 0.35 if desc is None else 0.0          # catch-all "other"
                lg = self.sharpness * _overlap(keys, sw) + base + rng.gauss(0, self.noise) + jitter()
                if name.lower() in injected:
                    lg += 4.0
                logits.append(lg)
            ps = _softmax(logits)
            probs = {n: round(p, 2) for n, p in zip(names, ps)}
            best = max(probs, key=probs.get)
            return {"type": "choice", "choice": best, "probabilities": probs,
                    "confidence": round(_peakedness(list(probs.values())), 2)}

        if kind == "score":
            levels = list(q["criteria"])
            logits = [self.sharpness * _overlap(_words(lv), sw) + rng.gauss(0, self.noise) + jitter()
                      for lv in levels]
            ps = _softmax(logits)
            probs = {str(i): round(p, 2) for i, p in enumerate(ps)}
            score = sum(i * p for i, p in enumerate(probs.values()))
            return {"type": "score", "score": round(score, 2),
                    "legend": {str(i): lv for i, lv in enumerate(levels)},
                    "probabilities": probs,
                    "confidence": round(_peakedness(list(probs.values())), 2)}
        raise ValueError(f"unknown question type {kind}")
