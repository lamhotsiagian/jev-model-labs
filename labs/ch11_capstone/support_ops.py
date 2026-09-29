"""Chapter 10 capstone -- a support operations platform built on every chapter.

    ingest ticket
      -> guard + serialize state                      (Ch. 4, 9)
      -> ONE fan-out call: 15 atomic questions        (Ch. 5)
      -> extract refund order ID via candidates       (Ch. 6)
      -> per-action gates scaled to cost              (Ch. 3, 5)
      -> LLM drafts a reply; Jev verifies each claim  (Ch. 7)
      -> decision log, audit sample, drift inputs     (Ch. 8)

Run: python -m labs.ch11_capstone.support_ops [--n 200]
Writes labs/ch11_capstone/out/{decisions.jsonl, summary.json}.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from jevkit import JevClient, NoulQ
from jevkit.contract import NoulAnswer
from jevkit.datasets import make_tickets
from jevkit.helpers import select_span
from jevkit.policy import Outcome, Policy
from jevkit.registry import QuestionRegistry
from jevkit.security import CANARY_Q, build_guarded_state
from jevkit.telemetry import DecisionLog, DecisionLogger, now

HERE = Path(__file__).parent
OUT = HERE / "out"
REG = QuestionRegistry.from_yaml(HERE / "questions.yaml")
POLICY = Policy.from_yaml(HERE / "policy.yaml")
ORDER_RX = re.compile(r"\bA-\d{3}\b")
BINDINGS = {"department": "assign_queue", "checkout_outage": "page_oncall",
            "refund_requested": "offer_refund", "churn_threat": "flag_churn_risk"}
REPLY_TEMPLATES = {  # the LLM is stubbed with templates so the lab runs offline
    "billing": "We are sorry about the charge on order {order}. A refund has been initiated and "
               "will appear within 5 business days.",
    "technical": "Our engineers are investigating the error in the integration and API you reported; "
                 "updates on the outage will be posted on the status page.",
    "sales": "A member of our sales team will send pricing options today.",
    "account": "We have sent a password reset link so you can login and regain access to your account.",
}


@dataclass
class TicketOutcome:
    ticket_id: str
    actions: dict[str, str] = field(default_factory=dict)
    refund_order: str | None = None
    priority: float = 0.0
    reply_sent: bool = False
    suspicious: bool = False


class SupportOps:
    def __init__(self, client: JevClient, logger: DecisionLogger):
        self.client, self.logger = client, logger
        self.questions = {**REG.questions(), "injection_canary": CANARY_Q}

    def handle(self, ticket) -> TicketOutcome:
        out = TicketOutcome(ticket.id)
        state, scan = build_guarded_state({"ticket_id": ticket.id, "plan": "pro"}, ticket.text)
        res = self.client.ask(state, self.questions)                       # ONE call, 16 questions
        out.suspicious = scan.suspicious or res["injection_canary"].p_true >= 0.5
        out.priority = POLICY.composite("priority", res.answers)

        for qid, action in BINDINGS.items():
            d = POLICY.decide(action, res[qid])
            outcome = d.outcome
            if out.suspicious and outcome is Outcome.ACT:                   # governance downgrade
                outcome = Outcome.CONFIRM
            out.actions[action] = outcome.value
            self._log(ticket.id, qid, action, res, outcome.value)

        if out.actions["offer_refund"] == "act" and res["mentions_order"].p_true >= 0.5:
            span = select_span(self.client, ticket.text, ORDER_RX, "the order the customer wants refunded")
            out.refund_order = span.value if span.confidence >= 0.5 else None
        if out.actions["assign_queue"] == "act":
            out.reply_sent = self._draft_and_verify(res["department"].choice, out.refund_order, ticket.text)
        out.actions["send_draft_reply"] = "act" if out.reply_sent else "escalate"
        return out

    def _draft_and_verify(self, team: str, order: str | None, text: str) -> bool:
        """Consistency check: classify the DRAFT with the same department Choice
        and require it to land on the ticket's team. A second Noul detects
        promises (money, credit, deadlines), allowed only in billing replies."""
        if team == "billing" and not order:
            return False                                   # never send a refund reply without an order
        draft = REPLY_TEMPLATES[team].format(order=order)
        v = self.client.ask(draft, {
            "draft_topic": REG["department"].to_question(),
            "promises": NoulQ("Does the reply promise a refund, a credit, or a time frame in days?")})
        d = POLICY.decide("send_draft_reply", v["draft_topic"])
        on_topic = d.outcome is Outcome.ACT and v["draft_topic"].choice == team
        return on_topic and (v["promises"].p_true < 0.5 or team == "billing")

    def _log(self, tid, qid, action, res, outcome) -> None:
        a = res[qid]
        self.logger.log(DecisionLog(
            ts=now(), workflow=REG.workflow, question=REG[qid].versioned_id, model_version=res.model_version,
            kind="noul" if isinstance(a, NoulAnswer) else "choice",
            probabilities={"true": a.p_true} if isinstance(a, NoulAnswer) else dict(a.probabilities),
            confidence=None if isinstance(a, NoulAnswer) else a.confidence, outcome=outcome,
            latency_ms=res.latency_ms, input_tokens=res.input_tokens, request_id=tid,
            extra={"action": action, "policy_version": POLICY.version}))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    (OUT / "decisions.jsonl").unlink(missing_ok=True)
    ops = SupportOps(JevClient(), DecisionLogger(OUT / "decisions.jsonl", audit_rate=0.03, seed=0))
    tickets = make_tickets(args.n, seed=2026)
    outcomes = [ops.handle(t) for t in tickets]

    tally = {a: Counter(o.actions[a] for o in outcomes) for a in outcomes[0].actions}
    refunds = [(t, o) for t, o in zip(tickets, outcomes) if o.refund_order]
    right_refund = sum(o.refund_order == t.refund_order for t, o in refunds)
    queue_acts = [(t, o) for t, o in zip(tickets, outcomes) if o.actions["assign_queue"] == "act"]
    summary = {
        "tickets": len(tickets), "questions_per_call": len(ops.questions),
        "actions": {a: dict(c) for a, c in tally.items()},
        "resolved_without_human": round(sum(o.reply_sent and "confirm" not in o.actions.values()
                                            for o in outcomes) / len(outcomes), 3),
        "refund_orders_extracted": len(refunds), "refund_order_correct": right_refund,
        "replies_sent": sum(o.reply_sent for o in outcomes),
        "flagged_suspicious": sum(o.suspicious for o in outcomes),
        "decision_log_lines": sum(1 for _ in (OUT / "decisions.jsonl").open()),
        "backend": "SIMULATED" if ops.client.is_simulated else "MEASURED",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
