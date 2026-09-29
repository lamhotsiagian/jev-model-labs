"""Chapter 8 lab -- the operations stack.

Flow:
  1. pinning + golden-set gate: evaluate the pinned version and a candidate
     version on the same labelled golden set; block the upgrade on regression;
  2. decision logging: one JSONL line per answer, with audit sampling;
  3. drift: baseline week vs. a week where a new customer segment arrives
     (new vocabulary) -> PSI and escalation-rate change per question;
  4. audit loop: human labels on sampled auto-acts flow back into the labelled set;
  5. weekly calibration report (the job a scheduler runs every Monday).

Run: python -m labs.ch08_ops.ops_lab
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from jevkit import JevClient
from jevkit.calibration import ece
from jevkit.contract import NoulAnswer
from jevkit.datasets import make_tickets
from jevkit.policy import Policy
from jevkit.registry import QuestionRegistry
from jevkit.simulator import JevSimulator
from jevkit.telemetry import DecisionLog, DecisionLogger, drift_report, golden_set_gate, now

HERE = Path(__file__).parent
OUT = HERE / "out"
REG = QuestionRegistry.from_yaml(HERE.parent / "ch04_state" / "questions.yaml")
POLICY = Policy.from_yaml(HERE.parent / "ch05_policy" / "policy.yaml")
BIND = {"department": "assign_queue", "checkout_outage": "page_oncall", "refund_requested": "offer_refund"}


def golden(client: JevClient, tickets) -> dict[str, bool]:
    q = REG.questions(["department"])
    return {t.id: client.ask(t.text, q)["department"].choice == t.department for t in tickets}


def run_week(client: JevClient, tickets, logger: DecisionLogger) -> list[dict]:
    rows = []
    for t in tickets:
        res = client.ask(t.text, REG.questions(list(BIND)))
        for qid, action in BIND.items():
            a, d = res[qid], POLICY.decide(action, res[qid])
            rec = logger.log(DecisionLog(
                ts=now(), workflow="support_triage", question=REG[qid].versioned_id,
                model_version=res.model_version, kind="noul" if isinstance(a, NoulAnswer) else "choice",
                probabilities={"true": a.p_true} if isinstance(a, NoulAnswer) else dict(a.probabilities),
                confidence=None if isinstance(a, NoulAnswer) else a.confidence, outcome=d.outcome.value,
                latency_ms=res.latency_ms, input_tokens=res.input_tokens, request_id=t.id,
                extra={"label": t.department if qid == "department" else None}))
            rows.append(json.loads(json.dumps(rec.__dict__)))
    return rows


def shifted_segment(n: int, seed: int):
    """A new enterprise segment writes differently: procurement jargon, no keywords we tuned on."""
    rng = random.Random(seed)
    ts = make_tickets(n, seed=seed)
    jargon = ["per our MSA", "procurement flagged", "PO reference attached", "per SOW clause 4",
              "vendor onboarding portal", "legal review pending"]
    for t in ts:
        words = t.text.split()
        keep = [w for w in words if rng.random() > 0.45]           # terser messages
        t.text = " ".join(keep + [rng.choice(jargon), rng.choice(jargon)])
    return ts


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for f in OUT.glob("*.jsonl"):
        f.unlink()
    pinned = JevClient(JevSimulator(version="jev-1.13.0-sim"))
    candidate = JevClient(JevSimulator(version="jev-1.14.0-sim", sharpness=3.4, noise=0.5))
    gold = make_tickets(300, seed=99)

    print("== 1. golden-set upgrade gate ==")
    gate = golden_set_gate(golden(pinned, gold), golden(candidate, gold), max_regression=0.01)
    print(json.dumps(gate, indent=1))
    print("-> upgrade", "APPROVED" if gate["pass"] else "BLOCKED: keep jev-1.13.0 pinned")

    print("\n== 2-3. logging and drift ==")
    base = run_week(pinned, make_tickets(300, seed=1), DecisionLogger(OUT / "week1.jsonl", audit_rate=0.25, seed=1))
    cur = run_week(pinned, shifted_segment(300, seed=2), DecisionLogger(OUT / "week2.jsonl", audit_rate=0.25, seed=2))
    for row in drift_report(base, cur):
        print(row)

    print("\n== 4. audit loop ==")
    audited = [r for r in cur if r["audited"]]
    disagreements = [r for r in audited if r["extra"].get("label") and
                     max(r["probabilities"], key=r["probabilities"].get) != r["extra"]["label"]]
    # Lab uses a 25% audit rate so the sample is visible; production runs 1-5%.
    print(f"audited auto-acts: {len(audited)}; disagreements to add to the labelled set: {len(disagreements)}")

    print("\n== 5. weekly calibration report ==")
    dept = [r for r in cur if r["question"].startswith("department")]
    conf = [r["confidence"] for r in dept]
    correct = [max(r["probabilities"], key=r["probabilities"].get) == r["extra"]["label"] for r in dept]
    week1 = [r for r in base if r["question"].startswith("department")]
    report = {"week1_ece": round(ece([r["confidence"] for r in week1],
                                     [max(r["probabilities"], key=r["probabilities"].get) == r["extra"]["label"]
                                      for r in week1]), 3),
              "week2_ece": round(ece(conf, correct), 3),
              "week2_accuracy": round(sum(correct) / len(correct), 3)}
    print(report)
    (OUT / "weekly_report.json").write_text(json.dumps({"gate": gate, "calibration": report}, indent=2))


if __name__ == "__main__":
    main()
