"""Chapter 3 lab -- confidence toolkit and calibration study.

Flow:
  1. generate (or load) 400 labelled tickets;
  2. ask Jev the department Choice for each, keep the full distribution;
  3. compare three confidence measures (peakedness, top-2 margin, entropy)
     by how well they predict correctness (AUROC) and by ECE;
  4. draw the reliability diagram and the automation-vs-error curve;
  5. recommend a threshold for a given error budget.

Run: python -m labs.ch03_calibration.run_calibration [--error-budget 0.05]
Writes labs/ch03_calibration/out/{calibration.json, reliability.png, tradeoff.png}.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from jevkit import ChoiceQ, JevClient
from jevkit.calibration import (IsotonicCalibrator, Record, brier_multiclass, compare_measures, ece,
                                recommend_threshold, recommend_threshold_conservative,
                                reliability_bins, threshold_sweep)
from jevkit.confidence import MEASURES, peakedness, verify_against_returned
from jevkit.datasets import DEPARTMENTS, make_tickets

OUT = Path(__file__).parent / "out"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--error-budget", type=float, default=0.05)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    client = JevClient()
    q = {"department": ChoiceQ("Which team should handle this?", DEPARTMENTS)}
    records, returned = [], []
    for t in make_tickets(args.n):
        a = client.ask(t.text, q)["department"]
        records.append(Record(a.probabilities, t.department))
        returned.append((a.raw_probabilities, a.confidence))

    print("peakedness vs returned confidence:", verify_against_returned(returned))
    acc = sum(r.correct for r in records) / len(records)
    conf = [peakedness(r.probabilities) for r in records]
    pmax = [max(r.probabilities.values()) for r in records]
    correct = [r.correct for r in records]
    summary = {
        "backend": "SIMULATED" if client.is_simulated else "MEASURED",
        "n": len(records), "accuracy": round(acc, 3),
        "brier": round(brier_multiclass(records), 4),
        "ece_top_probability": round(ece(pmax, correct), 4),
        "measures": {k: {m: round(v, 4) for m, v in d.items()} for k, d in compare_measures(records, MEASURES).items()},
    }
    sweep = threshold_sweep(conf, correct)
    rec = recommend_threshold(sweep, args.error_budget)
    summary["recommendation"] = rec
    # Point estimates flatter small samples; certify on the Wilson upper bound instead.
    summary["recommendation_wilson"] = {
        str(b): recommend_threshold_conservative(conf, correct, b) for b in (args.error_budget, 2 * args.error_budget)}

    # Recalibration: fit on the first half, evaluate ECE of the mapped value on the second half.
    half = len(records) // 2
    iso = IsotonicCalibrator().fit(conf[:half], correct[:half])
    mapped = iso.predict(conf[half:])
    summary["recalibration"] = {"ece_raw_peakedness_holdout": round(ece(conf[half:], correct[half:]), 4),
                                "ece_isotonic_holdout": round(ece(mapped, correct[half:]), 4)}
    summary["reliability"] = reliability_bins(pmax, correct)
    (OUT / "calibration.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "reliability"}, indent=2))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        rows = summary["reliability"]
        fig, ax = plt.subplots(figsize=(4.2, 4))
        ax.plot([0, 1], [0, 1], "--", color="grey", label="perfect calibration")
        ax.plot([r["mean_conf"] for r in rows], [r["accuracy"] for r in rows], "o-", label="observed")
        ax.set(xlabel="mean top probability in bin", ylabel="accuracy in bin", title="Reliability diagram")
        ax.legend()
        fig.tight_layout(); fig.savefig(OUT / "reliability.png", dpi=160)
        fig, ax = plt.subplots(figsize=(5, 3.4))
        ax.plot([r["threshold"] for r in sweep], [r["automation"] for r in sweep], label="automation rate")
        ax.plot([r["threshold"] for r in sweep], [r["error_rate"] for r in sweep], label="error rate (acted)")
        if rec:
            ax.axvline(rec["threshold"], color="red", ls=":", label=f"recommended t={rec['threshold']}")
        ax.set(xlabel="confidence threshold", title="Automation vs error")
        ax.legend(); fig.tight_layout(); fig.savefig(OUT / "tradeoff.png", dpi=160)
    except ImportError:
        pass


if __name__ == "__main__":
    main()
