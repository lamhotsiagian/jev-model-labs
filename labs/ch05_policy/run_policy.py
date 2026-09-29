"""Chapter 5 lab -- the decision layer as code.

Flow:
  1. speculative fan-out: every question the workflow MIGHT need, in one call,
     compared against sequential calls (tokens and modelled latency);
  2. confidence-gated routing with per-action thresholds from policy.yaml;
  3. composite priority score with weights from config;
  4. intent routing: a Choice selects the handler before expensive work;
  5. expected-cost view: derive the break-even threshold from costs.

Run: python -m labs.ch05_policy.run_policy
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from jevkit import JevClient
from jevkit.datasets import make_tickets
from jevkit.policy import Outcome, Policy, breakeven_threshold, route_intent
from jevkit.registry import QuestionRegistry

HERE = Path(__file__).parent
REG = QuestionRegistry.from_yaml(HERE.parent / "ch04_state" / "questions.yaml")
POLICY = Policy.from_yaml(HERE / "policy.yaml")


def fanout_vs_sequential(client: JevClient, text: str) -> None:
    qs = REG.questions()
    one = client.ask(text, qs)
    seq_tokens = seq_ms = 0.0
    for qid, q in qs.items():
        r = client.ask(text, {qid: q})
        seq_tokens += r.input_tokens
        seq_ms += r.latency_ms or 0
    print(f"fan-out   : 1 call,  {one.input_tokens:5d} input tokens, {one.latency_ms:6.1f} ms")
    print(f"sequential: {len(qs)} calls, {int(seq_tokens):5d} input tokens, {seq_ms:6.1f} ms "
          f"-> {seq_tokens / one.input_tokens:.1f}x tokens, {seq_ms / one.latency_ms:.1f}x latency")


def main() -> None:
    client = JevClient()
    tickets = make_tickets(200, seed=21)
    print("== 1. speculative fan-out ==")
    fanout_vs_sequential(client, tickets[0].text)

    print("\n== 2. per-action gating over 200 tickets ==")
    tally: dict[str, Counter] = {a: Counter() for a in ("assign_queue", "auto_close_ticket", "page_oncall")}
    wrong_acts = Counter()
    for t in tickets:
        res = client.ask(t.text, REG.questions())
        d = POLICY.decide("assign_queue", res["department"])
        tally["assign_queue"][d.outcome.value] += 1
        if d.outcome is Outcome.ACT and res["department"].choice != t.department:
            wrong_acts["assign_queue"] += 1
        tally["auto_close_ticket"][POLICY.decide("auto_close_ticket", res["department"]).outcome.value] += 1
        p = POLICY.decide("page_oncall", res["checkout_outage"])
        tally["page_oncall"][p.outcome.value] += 1
        if p.outcome is Outcome.ACT and not t.outage:
            wrong_acts["page_oncall"] += 1
    for a, c in tally.items():
        print(f"{a:18s} act={c['act']:3d} confirm={c['confirm']:3d} escalate={c['escalate']:3d} "
              f"wrong_acts={wrong_acts[a]}")

    print("\n== 3. composite priority (weights from YAML) ==")
    for t in tickets[:5]:
        res = client.ask(t.text, REG.questions(["frustration", "churn_threat", "checkout_outage"]))
        print(f"{t.id} priority={POLICY.composite('priority', res.answers):.2f}  | {t.text[:70]}")

    print("\n== 4. intent routing ==")
    handlers = {"billing": lambda: "billing_db_lookup", "technical": lambda: "status_page_check",
                "sales": lambda: "crm_quote_flow", "account": lambda: "identity_reset_flow"}
    for t in tickets[:4]:
        ans = client.ask(t.text, REG.questions(["department"]))["department"]
        print(f"{t.id} -> {route_intent(ans, handlers, 0.5, lambda: 'human_triage'):22s} "
              f"(choice={ans.choice}, conf={ans.confidence:.2f})")

    print("\n== 5. break-even thresholds from cost ==")
    for name, ap in POLICY.actions.items():
        print(f"{name:18s} configured act_at={ap.act_at:.2f}  break-even={breakeven_threshold(ap.cost_false_positive, ap.cost_escalation):.2f}")


if __name__ == "__main__":
    main()
