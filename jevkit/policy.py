"""jevkit.policy -- the decision layer: Jev returns beliefs, your code owns policy (Chapter 5).

Four documented patterns, implemented as code rather than prompt text:

1. Speculative fan-out   ask every independent question you might need in ONE call.
2. Confidence-gated routing  act / confirm / escalate bands, per action.
3. Composite scoring     normalize Scores, weight them in config, combine in code.
4. Intent routing        a Choice picks the handler before expensive work runs.

The central practice: thresholds are per ACTION, scaled to the cost of being
wrong, and they live in YAML owned by the product team, not inside questions.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml

from .confidence import noul_decisiveness
from .contract import Answer, ChoiceAnswer, NoulAnswer, ScoreAnswer


class Outcome(str, Enum):
    ACT = "act"
    CONFIRM = "confirm"        # act only after a cheap human or secondary check
    ESCALATE = "escalate"      # hand to a person, do nothing automatically


@dataclass(frozen=True)
class ActionPolicy:
    action: str
    act_at: float                      # confidence (or probability) needed to act
    confirm_at: float | None = None    # optional middle band
    signal: str = "confidence"         # confidence | p_true | decisiveness
    cost_false_positive: float = 1.0   # documentation + expected-cost analysis
    cost_escalation: float = 0.1

    def __post_init__(self) -> None:
        if not 0.0 <= self.act_at <= 1.0:
            raise ValueError(f"{self.action}: act_at must be in [0,1]")
        if self.confirm_at is not None and self.confirm_at > self.act_at:
            raise ValueError(f"{self.action}: confirm_at must be <= act_at")


@dataclass(frozen=True)
class Decision:
    action: str
    outcome: Outcome
    signal_value: float
    threshold: float
    reason: str


class Policy:
    def __init__(self, actions: Mapping[str, ActionPolicy], weights: Mapping[str, Mapping[str, float]] | None = None,
                 version: str = "0"):
        self.actions = dict(actions)
        self.weights = {k: dict(v) for k, v in (weights or {}).items()}
        self.version = version

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Policy":
        doc = yaml.safe_load(Path(path).read_text())
        acts = {name: ActionPolicy(action=name, **cfg) for name, cfg in doc["actions"].items()}
        return cls(acts, doc.get("composites", {}), str(doc.get("version", "0")))

    # -- pattern 2: confidence-gated routing ---------------------------------
    def signal(self, ap: ActionPolicy, ans: Answer) -> float:
        if isinstance(ans, NoulAnswer):
            if ap.signal == "decisiveness":
                return noul_decisiveness(ans.p_true)
            return ans.p_true          # a Noul's natural signal is P(yes)
        if ap.signal == "confidence":
            return ans.confidence      # type: ignore[union-attr]
        raise ValueError(f"{ap.action}: signal {ap.signal} invalid for {type(ans).__name__}")

    def decide(self, action: str, ans: Answer) -> Decision:
        if action not in self.actions:
            raise KeyError(f"no policy for action '{action}' (fail closed)")
        ap = self.actions[action]
        v = self.signal(ap, ans)
        if v >= ap.act_at:
            return Decision(action, Outcome.ACT, v, ap.act_at, f"{ap.signal}={v:.2f} >= {ap.act_at}")
        if ap.confirm_at is not None and v >= ap.confirm_at:
            return Decision(action, Outcome.CONFIRM, v, ap.confirm_at,
                            f"{ap.confirm_at} <= {ap.signal}={v:.2f} < {ap.act_at}")
        return Decision(action, Outcome.ESCALATE, v, ap.act_at, f"{ap.signal}={v:.2f} below band")

    # -- pattern 3: composite scoring -----------------------------------------
    def composite(self, name: str, answers: Mapping[str, Answer]) -> float:
        """Weighted mean of normalized components in [0, 1].

        Score -> score / top_level;  Noul -> p_true;  Choice not allowed (no order).
        Weights come from YAML so a product owner can retune without touching
        question text, and without re-running the question-quality evaluation.
        """
        w = self.weights[name]
        num = den = 0.0
        for qid, weight in w.items():
            a = answers[qid]
            if isinstance(a, ScoreAnswer):
                x = a.normalized
            elif isinstance(a, NoulAnswer):
                x = a.p_true
            else:
                raise TypeError(f"{qid}: Choice has no order; cannot be a composite component")
            num += weight * x
            den += abs(weight)
        return num / den if den else 0.0


# -- pattern 4: intent routing --------------------------------------------------
def route_intent(ans: ChoiceAnswer, handlers: Mapping[str, Callable[[], Any]],
                 min_conf: float, fallback: Callable[[], Any]) -> Any:
    """Dispatch to a handler chosen by a Choice; low confidence goes to fallback."""
    if ans.confidence < min_conf or ans.choice not in handlers:
        return fallback()
    return handlers[ans.choice]()


# -- expected cost view (used to *derive* thresholds) ---------------------------
def breakeven_threshold(cost_false_positive: float, cost_escalation: float) -> float:
    """Act when P(correct) * 0 + (1 - P(correct)) * C_fp < C_esc.

    => act when P(correct) > 1 - C_esc / C_fp. Assumes a CALIBRATED signal;
    apply it to a calibrated probability, not to raw peakedness.
    """
    if cost_false_positive <= 0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - cost_escalation / cost_false_positive))
