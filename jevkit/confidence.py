"""jevkit.confidence -- confidence statistics over a returned distribution (Chapter 3).

Confidence is not a second model output. It is a statistic computed from the
probability distribution the answer already carries, and TypeSafe returns the
full distribution precisely so you can compute a different statistic if it
predicts correctness better on your data.

PROVENANCE: ``peakedness`` reproduces a community reverse-engineering of the
returned confidence (max probability rescaled so uniform -> 0, one-hot -> 1).
It is third-party, not official. Verify it against your own responses with
``verify_against_returned``.
"""
from __future__ import annotations

import math
from typing import Mapping


def peakedness(p: Mapping[str, float]) -> float:
    """Normalized max probability: uniform -> 0.0, one-hot -> 1.0 (community-derived)."""
    k, m = len(p), max(p.values())
    if k < 2:
        return 1.0
    return max(0.0, min(1.0, (k * m - 1) / (k - 1)))


def top2_margin(p: Mapping[str, float]) -> float:
    """Gap between the best and second-best option. Sensitive to near-ties."""
    vals = sorted(p.values(), reverse=True) + [0.0]
    return vals[0] - vals[1]


def entropy_conf(p: Mapping[str, float]) -> float:
    """1 - normalized Shannon entropy. Uses the whole distribution, not just the peak."""
    k = len(p)
    if k < 2:
        return 1.0
    h = -sum(v * math.log(v) for v in p.values() if v > 0)
    return max(0.0, 1.0 - h / math.log(k))


def noul_decisiveness(p_true: float) -> float:
    """A Noul has no confidence field. This is the binary analogue of peakedness:
    distance from 0.5 rescaled to [0, 1]. Use it for routing, never as P(correct)."""
    return abs(2.0 * p_true - 1.0)


MEASURES = {"peakedness": peakedness, "top2_margin": top2_margin, "entropy": entropy_conf}


def verify_against_returned(samples: list[tuple[Mapping[str, float], float]],
                            tol: float = 0.02) -> dict:
    """Compare our peakedness with the ``confidence`` Jev returned.

    ``samples`` is a list of (raw_probabilities, returned_confidence). Because
    both sides are rounded to 2 decimals we allow a small tolerance.
    """
    errs = [abs(peakedness(p) - c) for p, c in samples]
    within = sum(e <= tol for e in errs)
    return {"n": len(errs), "max_abs_err": max(errs, default=0.0),
            "share_within_tol": within / max(1, len(errs))}
