"""Contract tests run on recorded fixtures: they never call the API."""
import json
from pathlib import Path

import pytest

from jevkit.contract import (ChoiceQ, ContractError, NoulQ, ScoreQ, build_request,
                             check_score_monotonic, normalize, parse_response)

FIX = json.loads((Path(__file__).parent / "fixture_triage.json").read_text())
QS = {"department": ChoiceQ("Which team?", {k: v for k, v in
                                            FIX["request"]["questions"]["department"]["criteria"].items()}),
      "refund_requested": NoulQ("Refund?"),
      "frustration": ScoreQ("How frustrated?", FIX["request"]["questions"]["frustration"]["criteria"])}


def test_parse_typed_answers_and_version():
    res = parse_response(FIX["response"], QS)
    assert res.model_version.startswith("jev-")
    assert res.model_version != "jev-latest"          # concrete version is recorded
    assert 0.0 <= res["refund_requested"].p_true <= 1.0
    assert res["department"].choice in QS["department"].criteria
    assert abs(sum(res["department"].probabilities.values()) - 1.0) < 1e-9


def test_normalize_tolerates_rounding_but_rejects_garbage():
    assert abs(sum(normalize({"a": .33, "b": .33, "c": .33}).values()) - 1) < 1e-12
    with pytest.raises(ContractError):
        normalize({"a": .5, "b": .2})


def test_choice_limit_and_score_levels():
    with pytest.raises(ContractError):
        ChoiceQ("x", {f"o{i}": None for i in range(256)})
    with pytest.raises(ContractError):
        ScoreQ("x", ["only one"])
    with pytest.raises(ContractError):
        ScoreQ("x", [str(i) for i in range(11)])


def test_type_mismatch_is_caught():
    wrong = dict(QS, refund_requested=ChoiceQ("x", {"a": "a", "b": "b"}))
    with pytest.raises(ContractError):
        parse_response(FIX["response"], wrong)


def test_request_body_shape():
    body = build_request("s", {"q": NoulQ("i")})
    assert body == {"model": "jev-latest", "state": "s", "questions": {"q": {"type": "noul", "instructions": "i"}}}


def test_score_rubric_order_lint():
    assert check_score_monotonic(["calm", "frustrated", "angry"], ["calm", "frustrated", "angry"]) == []
    assert check_score_monotonic(["angry", "calm"], ["calm", "frustrated", "angry"])
