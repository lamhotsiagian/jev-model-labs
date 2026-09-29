"""jevkit.contract -- the typed request/response contract for Jev (Chapter 2).

Jev exposes one endpoint (POST /v1/systemone), three question primitives
(Noul, Choice, Score) and one response shape. This module turns that raw JSON
into frozen dataclasses your application can trust, and it enforces the
details that bite in production:

* probabilities arrive rounded to two decimals, so they may not sum to 1.0;
* Choice criteria are a *map* (option -> description), Score criteria are an
  *ordered array* from low to high, and the response returns a ``legend``
  mapping indices back to level text;
* a Noul answer is a single yes-probability with no separate confidence field;
* the request names ``jev-latest`` but the response names the concrete version
  (for example ``jev-1.13.0``); we record it on every answer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

MAX_CHOICE_OPTIONS = 255          # documented Choice limit
MIN_SCORE_LEVELS, MAX_SCORE_LEVELS = 2, 10
ROUNDING_TOLERANCE = 0.05         # 2-dp rounding over up to ~10 buckets


class ContractError(ValueError):
    """Raised when a request or response violates the Jev contract."""


# ---------------------------------------------------------------------------
# Request side: question definitions
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class NoulQ:
    instructions: Any
    criteria: Mapping[str, str] | None = None   # optional {"true": ..., "false": ...}

    def to_wire(self) -> dict:
        body = {"type": "noul", "instructions": self.instructions}
        if self.criteria:
            body["criteria"] = dict(self.criteria)
        return body


@dataclass(frozen=True)
class ChoiceQ:
    instructions: Any
    criteria: Mapping[str, str | None]          # option name -> description

    def __post_init__(self) -> None:
        n = len(self.criteria)
        if n < 2:
            raise ContractError("Choice needs at least two options")
        if n > MAX_CHOICE_OPTIONS:
            raise ContractError(
                f"Choice has {n} options; the limit is {MAX_CHOICE_OPTIONS}. "
                "Use jevkit.helpers.hierarchical_classify instead.")

    def to_wire(self) -> dict:
        return {"type": "choice", "instructions": self.instructions,
                "criteria": dict(self.criteria)}


@dataclass(frozen=True)
class ScoreQ:
    instructions: Any
    criteria: Sequence[str]                     # ordered LOW -> HIGH

    def __post_init__(self) -> None:
        n = len(self.criteria)
        if not MIN_SCORE_LEVELS <= n <= MAX_SCORE_LEVELS:
            raise ContractError(f"Score needs 2-10 levels, got {n}")
        if len(set(self.criteria)) != n:
            raise ContractError("Score levels must be distinct")

    def to_wire(self) -> dict:
        return {"type": "score", "instructions": self.instructions,
                "criteria": list(self.criteria)}


Question = NoulQ | ChoiceQ | ScoreQ


def build_request(state: Any, questions: Mapping[str, Question],
                  model: str = "jev-latest") -> dict:
    """Serialize a request body exactly as the /v1/systemone endpoint expects."""
    if not questions:
        raise ContractError("at least one question is required")
    return {"model": model, "state": state,
            "questions": {qid: q.to_wire() for qid, q in questions.items()}}


# ---------------------------------------------------------------------------
# Response side: typed answers
# ---------------------------------------------------------------------------
def normalize(probs: Mapping[str, float]) -> dict[str, float]:
    """Renormalize 2-dp rounded probabilities so they sum to exactly 1.0.

    Rounding can make {a: .33, b: .33, c: .33} sum to .99. Downstream math
    (entropy, expected cost, composite scores) assumes a proper distribution,
    so we fix it once, here, and never again.
    """
    total = sum(probs.values())
    if total <= 0:
        raise ContractError("probability mass is zero")
    if abs(total - 1.0) > ROUNDING_TOLERANCE:
        raise ContractError(f"probabilities sum to {total:.3f}; not a rounding artefact")
    return {k: v / total for k, v in probs.items()}


@dataclass(frozen=True)
class Meta:
    question_id: str
    model_version: str              # concrete version, e.g. "jev-1.13.0"
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float | None = None


@dataclass(frozen=True)
class NoulAnswer:
    meta: Meta
    p_true: float                   # the estimated yes-probability; no confidence field

    @property
    def value(self) -> bool:
        return self.p_true >= 0.5


@dataclass(frozen=True)
class ChoiceAnswer:
    meta: Meta
    choice: str
    probabilities: Mapping[str, float]   # normalized
    confidence: float
    raw_probabilities: Mapping[str, float] = field(default_factory=dict)

    @property
    def top2_margin(self) -> float:
        a, b = (sorted(self.probabilities.values(), reverse=True) + [0.0])[:2]
        return a - b


@dataclass(frozen=True)
class ScoreAnswer:
    meta: Meta
    score: float                         # probability-weighted mean level index
    legend: Mapping[int, str]            # index -> level text
    probabilities: Mapping[str, float]   # keyed by level index as string
    confidence: float

    @property
    def n_levels(self) -> int:
        return len(self.legend)

    @property
    def normalized(self) -> float:
        """Score rescaled to [0, 1] by dividing by the top level index."""
        return self.score / (self.n_levels - 1)

    @property
    def argmax_level(self) -> str:
        idx = max(self.probabilities, key=self.probabilities.get)
        return self.legend[int(idx)]


Answer = NoulAnswer | ChoiceAnswer | ScoreAnswer


@dataclass(frozen=True)
class JevResult:
    model_version: str
    answers: Mapping[str, Answer]
    input_tokens: int
    output_tokens: int
    latency_ms: float | None = None

    def __getitem__(self, qid: str) -> Answer:
        return self.answers[qid]


def parse_response(raw: Mapping[str, Any], questions: Mapping[str, Question] | None = None,
                   latency_ms: float | None = None) -> JevResult:
    """Convert a raw /v1/systemone JSON body into typed answers.

    When ``questions`` is supplied we also check that every answer's type
    matches the question we asked and that Choice labels are ones we offered.
    Type safety is guaranteed by Jev's sampler; this check guards *our* code
    (wrong fixture, wrong question ID after a registry refactor).
    """
    try:
        version = raw["model"]
        usage = raw.get("usage", {})
        answers_raw = raw["answers"]
    except KeyError as exc:
        raise ContractError(f"missing field {exc}") from exc

    in_tok, out_tok = int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0))
    answers: dict[str, Answer] = {}
    for qid, a in answers_raw.items():
        meta = Meta(qid, version, in_tok, out_tok, latency_ms)
        kind = a.get("type")
        if questions is not None:
            expected = questions[qid].to_wire()["type"]
            if kind != expected:
                raise ContractError(f"{qid}: asked {expected}, got {kind}")
        if kind == "noul":
            answers[qid] = NoulAnswer(meta, float(a["noul"]))
        elif kind == "choice":
            probs = {k: float(v) for k, v in a["probabilities"].items()}
            if questions is not None:
                offered = set(questions[qid].criteria)  # type: ignore[union-attr]
                if not set(probs) <= offered:
                    raise ContractError(f"{qid}: unknown labels {set(probs) - offered}")
            answers[qid] = ChoiceAnswer(meta, a["choice"], normalize(probs),
                                        float(a["confidence"]), probs)
        elif kind == "score":
            legend = {int(k): v for k, v in a["legend"].items()}
            probs = {str(k): float(v) for k, v in a["probabilities"].items()}
            answers[qid] = ScoreAnswer(meta, float(a["score"]), legend,
                                       normalize(probs), float(a["confidence"]))
        else:
            raise ContractError(f"{qid}: unknown answer type {kind!r}")
    return JevResult(version, answers, in_tok, out_tok, latency_ms)


def check_score_monotonic(levels: Sequence[str], severity_words: Sequence[str]) -> list[str]:
    """Lint a Score rubric: flag levels whose text is not ordered low -> high.

    A cheap heuristic: each level should mention a severity word at an index
    no lower than the previous level. It catches the classic bug of listing
    levels high -> low, which silently inverts every downstream threshold.
    """
    rank = {w.lower(): i for i, w in enumerate(severity_words)}
    problems, last = [], -1
    for lvl in levels:
        hits = [rank[w] for w in rank if w in lvl.lower()]
        if hits:
            cur = max(hits)
            if cur < last:
                problems.append(f"level '{lvl}' is less severe than its predecessor")
            last = max(last, cur)
    return problems
