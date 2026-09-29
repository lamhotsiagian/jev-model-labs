"""Chapter 4 lab -- state engineering and question design.

Flow:
  A. serializers: render a DB row, an event window and a document chunk;
  B. A/B test: prose dump of the whole row vs. allow-listed key-value state,
     same questions, same labelled tickets -> accuracy and tokens;
  C. decomposition: one holistic 'Is this urgent?' vs. atomic questions
     combined in code -> accuracy and run-to-run consistency under
     meaning-preserving perturbations of the state;
  D. registry lint + fingerprint.

Run: python -m labs.ch04_state.state_lab
"""
from __future__ import annotations

import json
import random
from datetime import datetime, timedelta
from pathlib import Path

from jevkit import ChoiceQ, JevClient, NoulQ
from jevkit.datasets import DEPARTMENTS, make_tickets
from jevkit.registry import QuestionRegistry, RegisteredQuestion
from jevkit.serializers import serialize_chunk, serialize_events, serialize_prose, serialize_row

HERE = Path(__file__).parent
OUT = HERE / "out"
REG = QuestionRegistry.from_yaml(HERE / "questions.yaml")
NOISE = ["legacy account migrated from old billing system", "sales rep note: renewal quote pending",
         "previous ticket about password login loop", "API key rotated last quarter",
         "customer on invoice terms net 30", "upgrade discussion paused"]


def to_row(t, rng) -> dict:
    """A realistic CRM row: the useful message plus fields the question never needs."""
    return {"ticket_id": t.id, "plan": rng.choice(["starter", "pro", "enterprise"]),
            "region": rng.choice(["us-west", "eu-central", "ap-south"]),
            "internal_notes": "; ".join(rng.sample(NOISE, 3)),
            "customer_message": t.text, "created": t.created.isoformat()}


def part_a() -> None:
    rng = random.Random(1)
    t = make_tickets(1)[0]
    print("--- serialize_row (allow-listed) ---")
    print(serialize_row(to_row(t, rng), fields=["plan", "customer_message"],
                        rename={"customer_message": "message"}))
    now = datetime(2026, 9, 28, 12, 0)
    evs = [{"ts": now - timedelta(minutes=m), "type": ty, "detail": d} for m, ty, d in
           [(42, "deploy", "checkout-svc v412"), (31, "alert", "5xx rate 7%"),
            (18, "ticket", "checkout error x3"), (2, "rollback", "checkout-svc v411")]]
    print("--- serialize_events ---\n" + serialize_events(evs, now=now))
    print("--- serialize_chunk ---\n" + serialize_chunk("The supplier may terminate on 30 days notice...",
                                                        "MSA-2291", "12. Termination", 7, 31)[:160])


def part_b(client: JevClient) -> dict:
    rng = random.Random(2)
    q = {"department": REG["department"].to_question()}
    res = {}
    tickets = make_tickets(300, seed=11)
    rows = [to_row(t, rng) for t in tickets]
    for name, fn in {"prose_full_row": serialize_prose,
                     "kv_allow_listed": lambda r: serialize_row(r, fields=["customer_message"],
                                                                rename={"customer_message": "message"})}.items():
        correct = toks = 0
        for t, row in zip(tickets, rows):
            r = client.ask(fn(row), q)
            correct += r["department"].choice == t.department
            toks += r.input_tokens
        res[name] = {"accuracy": round(correct / len(tickets), 3), "avg_input_tokens": round(toks / len(tickets))}
    print("B:", json.dumps(res))
    return res


def perturb(text: str, k: int) -> str:
    """Meaning-preserving variants: salutation swap, trailing signature, spacing."""
    sig = ["", " -- Sam", " Sent from my phone.", " Ref: web form", " Regards, Ops"][k % 5]
    return (text.replace("Hi team,", "Hello team,") if k % 2 else text) + sig


def part_c(client: JevClient) -> dict:
    holistic = {"urgent": NoulQ("Is this urgent?")}
    atomic_ids = ["checkout_outage", "churn_threat", "frustration"]
    atomic = REG.questions(atomic_ids)
    tickets = make_tickets(200, seed=5)
    truth = [t.outage or t.frustration == 2 for t in tickets]  # the business definition of 'priority'
    stats = {"holistic": [0, 0, 0], "atomic": [0, 0, 0]}   # correct, consistent, true positives
    for t, y in zip(tickets, truth):
        h_votes, a_votes = [], []
        for k in range(5):
            s = perturb(t.text, k)
            h_votes.append(client.ask(s, holistic)["urgent"].p_true >= 0.5)
            r = client.ask(s, atomic)
            a_votes.append(r["checkout_outage"].p_true >= 0.5 or r["churn_threat"].p_true >= 0.5
                           or r["frustration"].argmax_level.startswith("Very angry"))
        for name, votes in (("holistic", h_votes), ("atomic", a_votes)):
            stats[name][0] += votes[0] == y
            stats[name][1] += len(set(votes)) == 1
            stats[name][2] += votes[0] and y
    pos = sum(truth)
    out = {k: {"accuracy": round(v[0] / len(tickets), 3), "recall_on_priority": round(v[2] / pos, 3),
               "consistency_5_runs": round(v[1] / len(tickets), 3)} for k, v in stats.items()}
    print("C:", json.dumps(out))
    return out


def main() -> None:
    OUT.mkdir(exist_ok=True)
    client = JevClient()
    part_a()
    b, c = part_b(client), part_c(client)
    lint = REG.lint()
    print("D: registry", REG.workflow, "fingerprint", REG.fingerprint(), "questions", len(REG))
    print("   lint:", lint or "clean")
    bad = QuestionRegistry("demo", [RegisteredQuestion("urgent", "noul", 1, "nobody",
                                                       "Is this urgent and important?")])
    print("   lint of a deliberately bad question:", bad.lint())
    (OUT / "state_lab.json").write_text(json.dumps({"B": b, "C": c, "lint": lint}, indent=2))


if __name__ == "__main__":
    main()
