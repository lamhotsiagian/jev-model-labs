"""Chapter 10 lab -- JEV-as-a-judge with a confidence cascade.

Reproduces the METHOD of Li, Miao, Krishnan and Padman (2026, arXiv:2609.26550)
on synthetic pairwise items (the numbers are SIMULATED):

  1. judge 400 pairs in one order and in both orders (aligned averaging);
  2. accuracy and reversal inconsistency per category
     (ordinary, evidence-grounded, derivation, elaborate-wrong);
  3. accuracy by confidence bin q and AUROC of q for error detection;
  4. fit the escalation threshold tau on a 30% selection split;
  5. re-check tau on the 70% held-out split: accuracy retained, escalation
     rate, fee ratio against running the fallback judge on everything;
  6. per-category escalation: where the cascade spends its fallback budget.

Run: python -m labs.ch10_judge.judge_lab
Writes labs/ch10_judge/out/{judge.json, confidence_bins.png}.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from jevkit import JevClient
from jevkit.calibration import _auroc
from jevkit.datasets import make_judge_pairs
from jevkit.judge import SimulatedStrongJudge, evaluate_cascade, fee_ratio, fit_threshold, judge_pair
from jevkit.simulator import JevSimulator

OUT = Path(__file__).parent / "out"
BINS = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 0.95), (0.95, 1.001)]


def main() -> None:
    OUT.mkdir(exist_ok=True)
    # A noisier simulator setting stands in for a judge on harder, longer inputs.
    client = JevClient(JevSimulator(sharpness=2.5, noise=1.2))
    items = make_judge_pairs(400)
    single = {it.id: judge_pair(client, it, both_orders=False) for it in items}
    both = {it.id: judge_pair(client, it, both_orders=True) for it in items}

    print("== 1-2. accuracy and reversal inconsistency by category ==")
    by_cat = defaultdict(lambda: {"n": 0, "single": 0, "both": 0, "inconsistent": 0})
    for it in items:
        c = by_cat[it.category]
        c["n"] += 1
        c["single"] += single[it.id].choice == it.label
        c["both"] += both[it.id].choice == it.label
        c["inconsistent"] += not both[it.id].order_consistent
    cat_table = {k: {"single_order_acc": round(v["single"] / v["n"], 3),
                     "both_order_acc": round(v["both"] / v["n"], 3),
                     "reversal_inconsistency": round(v["inconsistent"] / v["n"], 3)}
                 for k, v in by_cat.items()}
    for k, v in cat_table.items():
        print(f"{k:16s} {v}")

    print("\n== 3. accuracy by confidence q (both orders) ==")
    q = np.array([both[i.id].q for i in items])
    ok = np.array([both[i.id].choice == i.label for i in items])
    bins = []
    for lo, hi in BINS:
        m = (q >= lo) & (q < hi)
        if m.any():
            bins.append({"bin": f"[{lo:.2f},{min(hi, 1):.2f}{']' if hi > 1 else ')'}",
                         "n": int(m.sum()), "acc": round(float(ok[m].mean()), 3)})
    for b in bins:
        print(b)
    auroc = _auroc(q, ok)
    print("error-detection AUROC of q:", round(auroc, 3))

    print("\n== 4-5. frozen cascade: fit tau on selection, re-check on held-out ==")
    fb = SimulatedStrongJudge()
    fb_choice = {it.id: fb(it) for it in items}
    cut = int(0.3 * len(items))
    sel, held = items[:cut], items[cut:]
    fit = fit_threshold(sel, both, fb_choice, max_drop=0.02)
    tau = fit["tau"]
    res = evaluate_cascade(held, both, fb_choice, tau)
    n_esc = sum(both[i.id].q < tau for i in held)
    fees = fee_ratio([both[i.id] for i in held], n_esc, len(held), fb.usd_per_1k)
    print("selection:", fit)
    print("held-out :", res)
    print("fees     :", fees)

    print("\n== 6. where the fallback budget goes ==")
    esc_cat = defaultdict(lambda: [0, 0])
    for it in held:
        esc_cat[it.category][0] += 1
        esc_cat[it.category][1] += both[it.id].q < tau
    esc_table = {k: round(v[1] / v[0], 3) for k, v in esc_cat.items()}
    print({k: f"{v:.0%} escalated" for k, v in esc_table.items()})
    # confident errors: the failure mode escalation cannot catch
    conf_wrong = defaultdict(int)
    for it in held:
        v = both[it.id]
        if v.q >= tau and v.choice != it.label:
            conf_wrong[it.category] += 1
    print("accepted-but-wrong by category:", dict(conf_wrong))

    print("\n== 7. envelope-aware cascade: out-of-envelope workloads go straight to the fallback ==")
    # Workload type is known from the task (a math or code task is identifiable), not from the label.
    OUT_OF_ENVELOPE = {"derivation", "elaborate_wrong"}
    sel_in = [i for i in sel if i.category not in OUT_OF_ENVELOPE]
    fit_in = fit_threshold(sel_in, both, fb_choice, max_drop=0.02)
    tau_in = fit_in["tau"]
    correct = esc = 0
    for it in held:
        if it.category in OUT_OF_ENVELOPE or both[it.id].q < tau_in:
            esc += 1
            correct += fb_choice[it.id] == it.label
        else:
            correct += both[it.id].choice == it.label
    fb_acc = sum(fb_choice[i.id] == i.label for i in held) / len(held)
    env = {"tau_in_envelope": tau_in, "cascade_acc": round(correct / len(held), 4),
           "fallback_acc": round(fb_acc, 4), "retained": round(correct / len(held) / fb_acc, 4),
           "escalation_rate": round(esc / len(held), 4),
           "fee_ratio": fee_ratio([both[i.id] for i in held], esc, len(held), fb.usd_per_1k)["fee_ratio"]}
    print(env)

    summary = {"backend": "SIMULATED", "envelope_cascade": env, "categories": cat_table, "confidence_bins": bins,
               "auroc_q": round(auroc, 3), "selection_fit": fit, "held_out": res, "fees": fees,
               "escalation_by_category": esc_table, "accepted_but_wrong": dict(conf_wrong)}
    (OUT / "judge.json").write_text(json.dumps(summary, indent=2))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 3.2))
        ax.bar([b["bin"] for b in bins], [b["acc"] for b in bins], color="#0E9488")
        ax.axhline(res["fallback_acc"], ls="--", color="grey", label="fallback accuracy (held-out)")
        ax.set(ylabel="Jev accuracy", xlabel="confidence q", ylim=(0, 1.05), title="Accuracy by confidence (SIMULATED)")
        ax.legend(); plt.xticks(rotation=30); fig.tight_layout()
        fig.savefig(OUT / "confidence_bins.png", dpi=160)
    except ImportError:
        pass


if __name__ == "__main__":
    main()
