"""jevkit.judge -- JEV-as-a-judge: accept when confident, escalate when unsure (Chapter 10).

Implements the evaluation pattern studied by Li, Miao, Krishnan and Padman (2026,
arXiv:2609.26550):

* pairwise preference judged with one Choice per request;
* both presentation orders, combined by aligned probability averaging
      p_bar(A) = 1/2 * [ p1(A first) + 1 - p2(B first) ]
* the decision's confidence q = max(p_bar, 1 - p_bar);
* a frozen cascade: accept Jev's verdict when q >= tau, otherwise escalate the
  item to a stronger (slower, costlier) fallback judge;
* tau fitted on a selection split, then re-checked on held-out items, because
  the paper found thresholds did not transfer for every fallback.

The fallback here is an interface: plug in any generative judge. The lab ships a
SIMULATED fallback whose per-category accuracy and fee are parameters, so the
cascade arithmetic can be exercised offline.
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

from .client import JevClient
from .contract import ChoiceQ

JUDGE_INSTRUCTIONS = ("Which response better completes the task? Judge factual and logical "
                      "correctness first, then helpfulness. Do not prefer a response because "
                      "it is longer or more elaborately written.")


@dataclass(frozen=True)
class PairItem:
    id: str
    task: str
    a: str
    b: str
    label: str                    # "A" or "B": the better response
    category: str                 # ordinary | evidence | derivation | elaborate_wrong
    evidence: str | None = None


@dataclass(frozen=True)
class Verdict:
    item_id: str
    choice: str                   # "A" or "B"
    p_a: float                    # aligned probability that A is better
    q: float                      # max(p_a, 1 - p_a): the escalation signal
    order_consistent: bool        # did both orders pick the same response?
    input_tokens: int
    calls: int


def _state(item: PairItem) -> dict:
    s = {"task": item.task}
    if item.evidence:
        s["evidence"] = item.evidence
    return s


def _ask_once(client: JevClient, item: PairItem, first: str, second: str) -> tuple[float, int]:
    """One request, one Choice. Returns P(first response is better) and tokens used.

    The candidates travel as the option descriptions, so each option carries the
    full text it stands for (the same idea as extraction by candidates).
    """
    q = ChoiceQ(JUDGE_INSTRUCTIONS, {"response_1": first, "response_2": second})
    res = client.ask(_state(item), {"pref": q})
    return res["pref"].probabilities["response_1"], res.input_tokens


def judge_pair(client: JevClient, item: PairItem, both_orders: bool = True) -> Verdict:
    p1, t1 = _ask_once(client, item, item.a, item.b)            # A shown first
    if not both_orders:
        p_a = p1
        return Verdict(item.id, "A" if p_a >= 0.5 else "B", p_a, max(p_a, 1 - p_a), True, t1, 1)
    p2, t2 = _ask_once(client, item, item.b, item.a)            # B shown first
    p_a = 0.5 * (p1 + (1.0 - p2))                               # aligned probability averaging
    consistent = (p1 >= 0.5) == (p2 < 0.5)
    return Verdict(item.id, "A" if p_a >= 0.5 else "B", p_a, max(p_a, 1 - p_a),
                   consistent, t1 + t2, 2)


# ---------------------------------------------------------------------------
# Fallback judges
# ---------------------------------------------------------------------------
FallbackJudge = Callable[[PairItem], str]     # returns "A" or "B"


@dataclass
class SimulatedStrongJudge:
    """Stand-in for a strong generative judge. NOT a model: a coin with per-category
    accuracy, seeded by item id, so cascade arithmetic is reproducible offline.
    Defaults echo the scale of the strongest comparator reported by Li et al. (2026)."""
    accuracy: Mapping[str, float] = field(default_factory=lambda: {
        "ordinary": 0.935, "evidence": 0.983, "derivation": 0.931, "elaborate_wrong": 0.946})
    usd_per_1k: float = 12.182
    calls: int = 0

    def __call__(self, item: PairItem) -> str:
        self.calls += 1
        h = int(hashlib.sha256(("fb" + item.id).encode()).hexdigest()[:12], 16)
        right = random.Random(h).random() < self.accuracy.get(item.category, 0.93)
        return item.label if right else ("B" if item.label == "A" else "A")


# ---------------------------------------------------------------------------
# Cascade
# ---------------------------------------------------------------------------
def cascade_decide(v: Verdict, tau: float, fallback: FallbackJudge, item: PairItem) -> tuple[str, bool]:
    """Accept Jev's verdict when q >= tau, else escalate. Returns (choice, escalated)."""
    if v.q >= tau:
        return v.choice, False
    return fallback(item), True


def evaluate_cascade(items: Sequence[PairItem], verdicts: Mapping[str, Verdict],
                     fallback_choices: Mapping[str, str], tau: float) -> dict:
    """Accuracy, escalation rate and accuracy retained relative to the fallback alone."""
    n = len(items)
    correct = esc = 0
    fb_correct = sum(fallback_choices[i.id] == i.label for i in items)
    for it in items:
        v = verdicts[it.id]
        if v.q >= tau:
            correct += v.choice == it.label
        else:
            esc += 1
            correct += fallback_choices[it.id] == it.label
    acc, fb_acc = correct / n, fb_correct / n
    return {"tau": tau, "n": n, "cascade_acc": round(acc, 4), "fallback_acc": round(fb_acc, 4),
            "escalation_rate": round(esc / n, 4), "retained": round(acc / fb_acc, 4) if fb_acc else 0.0}


def fit_threshold(items: Sequence[PairItem], verdicts: Mapping[str, Verdict],
                  fallback_choices: Mapping[str, str], max_drop: float = 0.02,
                  grid: Sequence[float] = tuple(round(0.5 + 0.05 * k, 2) for k in range(10)) + (0.99,)) -> dict:
    """Lowest tau (most items accepted) whose cascade accuracy on the SELECTION split
    stays within ``max_drop`` of the fallback's accuracy. Re-check on held-out items."""
    best = None
    for tau in sorted(grid, reverse=True):
        r = evaluate_cascade(items, verdicts, fallback_choices, tau)
        if r["cascade_acc"] >= r["fallback_acc"] - max_drop:
            best = r
    return best or evaluate_cascade(items, verdicts, fallback_choices, 1.01)


def fee_ratio(verdicts: Sequence[Verdict], escalated: int, n: int, fallback_usd_per_1k: float,
              jev_usd_per_mtok: float = 0.042) -> dict:
    """Cascade fee as a share of running the fallback on every item.
    Jev is paid on every item (it screens everything); the fallback only on escalations."""
    jev_usd = sum(v.input_tokens for v in verdicts) / 1e6 * jev_usd_per_mtok
    fb_usd = escalated * fallback_usd_per_1k / 1000
    all_fb = n * fallback_usd_per_1k / 1000
    return {"jev_usd": round(jev_usd, 5), "fallback_usd": round(fb_usd, 4),
            "cascade_usd": round(jev_usd + fb_usd, 4), "fallback_only_usd": round(all_fb, 4),
            "fee_ratio": round((jev_usd + fb_usd) / all_fb, 4) if all_fb else 0.0}


def to_json(obj) -> str:
    return json.dumps(obj, indent=2, default=lambda o: o.__dict__)
