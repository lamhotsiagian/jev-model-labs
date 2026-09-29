"""Chapter 2 lab -- the typed domain wrapper in action.

Flow:
  1. define a triage request with all three primitives;
  2. record the raw response as a fixture (so tests never hit the API);
  3. parse it into typed answers, normalize probabilities, check Score rubric order;
  4. map the answers onto a domain dataclass the rest of the system uses.

Run: python -m labs.ch02_contract.contract_demo
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from jevkit import ChoiceQ, JevClient, NoulQ, ScoreQ
from jevkit.client import RecordingBackend, default_backend
from jevkit.contract import build_request, check_score_monotonic
from jevkit.datasets import DEPARTMENTS, FRUSTRATION_LEVELS

FIXTURES = Path("data/fixtures")

QUESTIONS = {
    "department": ChoiceQ("Which team should handle this?", {**DEPARTMENTS, "other": None}),
    "refund_requested": NoulQ("Does the customer ask for money back?",
                              {"true": "Explicitly requests a refund or reversal",
                               "false": "No request for money back"}),
    "frustration": ScoreQ("How frustrated is the customer?", FRUSTRATION_LEVELS),
}


@dataclass(frozen=True)
class TriageView:
    """What the application sees. No raw dicts leak past this boundary."""
    team: str
    team_confidence: float
    wants_refund_p: float
    frustration_0_1: float
    model_version: str
    input_tokens: int


def to_view(res) -> TriageView:
    d, r, f = res["department"], res["refund_requested"], res["frustration"]
    return TriageView(d.choice, d.confidence, r.p_true, round(f.normalized, 3),
                      res.model_version, res.input_tokens)


def main() -> None:
    state = "I was charged twice for order A-104. Please refund the duplicate. This is getting frustrating."
    print("REQUEST BODY\n" + json.dumps(build_request(state, QUESTIONS), indent=2)[:900], "...\n")
    client = JevClient(RecordingBackend(default_backend(), FIXTURES))
    res = client.ask(state, QUESTIONS)
    for qid, a in res.answers.items():
        print(f"{qid:17s} {type(a).__name__:12s} {a}")
    raw = res["department"].raw_probabilities
    print(f"\nraw sum = {sum(raw.values()):.2f}  normalized sum = {sum(res['department'].probabilities.values()):.6f}")
    print("rubric lint:", check_score_monotonic(FRUSTRATION_LEVELS, ["calm", "frustrated", "angry"]) or "OK")
    print("reversed rubric lint:",
          check_score_monotonic(list(reversed(FRUSTRATION_LEVELS)), ["calm", "frustrated", "angry"]))
    print("\nDOMAIN VIEW:", to_view(res))


if __name__ == "__main__":
    main()
