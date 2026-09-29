"""jevkit.calibration -- is the model's probability honest on YOUR data? (Chapter 3)

RLCD trains Jev toward calibrated probabilities, but calibration is a property
of a model *on a distribution*. Your tickets, contracts or transactions are a
different distribution from TypeSafe's training data, so you measure it:

* Expected Calibration Error (ECE): weighted gap between stated confidence and
  observed accuracy across bins.
* Brier score: mean squared error of the probability against the 0/1 outcome.
* Reliability diagram: per-bin accuracy against per-bin mean probability.
* Threshold sweep: automation rate against error rate as the cut-off moves,
  which is the curve a product owner actually decides on.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class Record:
    """One labelled decision: the distribution Jev returned and the true label."""
    probabilities: Mapping[str, float]
    label: str

    @property
    def predicted(self) -> str:
        return max(self.probabilities, key=self.probabilities.get)

    @property
    def correct(self) -> bool:
        return self.predicted == self.label


def reliability_bins(conf: Sequence[float], correct: Sequence[bool], n_bins: int = 10):
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            rows.append({"lo": edges[b], "hi": edges[b + 1], "n": int(m.sum()),
                         "mean_conf": float(conf[m].mean()), "accuracy": float(correct[m].mean())})
    return rows


def ece(conf: Sequence[float], correct: Sequence[bool], n_bins: int = 10) -> float:
    rows, n = reliability_bins(conf, correct, n_bins), len(conf)
    return sum(r["n"] / n * abs(r["accuracy"] - r["mean_conf"]) for r in rows)


def brier_multiclass(records: Sequence[Record]) -> float:
    """Multi-class Brier: sum over classes of (p_k - y_k)^2, averaged over records."""
    total = 0.0
    for r in records:
        total += sum((p - (1.0 if k == r.label else 0.0)) ** 2 for k, p in r.probabilities.items())
    return total / len(records)


def brier_binary(p_true: Sequence[float], y: Sequence[bool]) -> float:
    p, y = np.asarray(p_true, float), np.asarray(y, float)
    return float(np.mean((p - y) ** 2))


def threshold_sweep(conf: Sequence[float], correct: Sequence[bool],
                    thresholds: Sequence[float] | None = None) -> list[dict]:
    """For each threshold t: act when conf >= t, escalate otherwise.

    automation = share of items acted on; error = wrong answers among acted items.
    """
    conf, correct = np.asarray(conf, float), np.asarray(correct, bool)
    ts = thresholds if thresholds is not None else np.round(np.arange(0.0, 1.0, 0.05), 2)
    out = []
    for t in ts:
        act = conf >= t
        n_act = int(act.sum())
        err = float((~correct[act]).mean()) if n_act else 0.0
        out.append({"threshold": float(t), "automation": n_act / len(conf),
                    "error_rate": err, "n_acted": n_act})
    return out


def recommend_threshold(sweep: list[dict], max_error: float) -> dict | None:
    """Lowest threshold (= most automation) whose error on acted items <= budget."""
    ok = [r for r in sweep if r["n_acted"] > 0 and r["error_rate"] <= max_error]
    return min(ok, key=lambda r: r["threshold"]) if ok else None


def compare_measures(records: Sequence[Record],
                     measures: Mapping[str, Callable[[Mapping[str, float]], float]]) -> dict:
    """Which confidence statistic best separates right from wrong answers?

    Reports AUROC of each measure as a predictor of correctness (0.5 = useless).
    """
    y = np.array([r.correct for r in records], bool)
    res = {}
    for name, fn in measures.items():
        s = np.array([fn(r.probabilities) for r in records])
        res[name] = {"auroc": _auroc(s, y), "ece": ece(s, y)}
    return res


def _auroc(score: np.ndarray, y: np.ndarray) -> float:
    pos, neg = score[y], score[~y]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # Mann-Whitney U with tie handling
    greater = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return float((greater + 0.5 * ties) / (len(pos) * len(neg)))


# ---------------------------------------------------------------------------
# Small-sample honesty: error-rate upper bounds and recalibration maps
# ---------------------------------------------------------------------------
def wilson_upper(errors: int, n: int, z: float = 1.645) -> float:
    """One-sided 95% Wilson upper bound on an error rate observed as errors/n.

    With 60 acted items and 2 errors the point estimate is 3.3%, but the true
    rate could plausibly be ~9%. Thresholds should be chosen on the bound.
    """
    if n == 0:
        return 1.0
    p = errors / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return min(1.0, (centre + margin) / denom)


def recommend_threshold_conservative(conf: Sequence[float], correct: Sequence[bool],
                                     max_error: float, min_acted: int = 30) -> dict | None:
    """Lowest threshold whose Wilson UPPER bound on acted-item error is within budget."""
    conf_a, corr_a = np.asarray(conf, float), np.asarray(correct, bool)
    best = None
    for t in sorted(set(np.round(conf_a, 2)), reverse=True):
        act = conf_a >= t
        n = int(act.sum())
        if n < min_acted:
            continue
        errs = int((~corr_a[act]).sum())
        ub = wilson_upper(errs, n)
        if ub <= max_error:
            best = {"threshold": float(t), "automation": n / len(conf_a), "n_acted": n,
                    "error_rate": errs / n, "error_upper_95": ub}
    return best


class IsotonicCalibrator:
    """Map a confidence statistic to an estimate of P(correct) (pool-adjacent-violators).

    Fit per question AND per model version on held-out labelled data; refit on
    every upgrade. The output is monotone, so it never reorders items: it only
    changes what the numbers mean, which is what thresholds need.
    """

    def fit(self, conf: Sequence[float], correct: Sequence[bool]) -> "IsotonicCalibrator":
        order = np.argsort(conf)
        x = np.asarray(conf, float)[order]
        y = np.asarray(correct, float)[order]
        blocks = [[yi, 1.0, xi, xi] for xi, yi in zip(x, y)]   # [sum_y, weight, x_lo, x_hi]
        merged: list[list[float]] = []
        for b in blocks:
            merged.append(b)
            while len(merged) > 1 and merged[-2][0] / merged[-2][1] > merged[-1][0] / merged[-1][1]:
                s2, w2, lo2, _ = merged.pop(-2)
                merged[-1] = [merged[-1][0] + s2, merged[-1][1] + w2, lo2, merged[-1][3]]
        self.x_hi = np.array([m[3] for m in merged])
        self.value = np.array([m[0] / m[1] for m in merged])
        return self

    def predict(self, conf: Sequence[float]) -> np.ndarray:
        idx = np.searchsorted(self.x_hi, np.asarray(conf, float), side="left")
        return self.value[np.clip(idx, 0, len(self.value) - 1)]
