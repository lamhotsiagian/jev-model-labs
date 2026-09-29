"""jevkit.cost -- a unit-economics model for decision workloads (Chapter 9).

Prices change; keep them in config and re-check the vendor page. Defaults
reflect the published early-access list price at the time of writing:
input $0.042 per million tokens, output tokens free.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Pricing:
    input_per_mtok: float = 0.042
    output_per_mtok: float = 0.0


@dataclass(frozen=True)
class Workload:
    decisions_per_day: int
    state_tokens: int
    questions_per_call: int
    tokens_per_question: int = 40
    cache_hit_rate: float = 0.0


def daily_cost(w: Workload, p: Pricing = Pricing(), fanout: bool = True) -> dict:
    """Compare one fan-out call per decision against one call per question.

    Sequential calls resend the state for every question; fan-out sends it once.
    That, not the per-token price, is where most of the saving comes from.
    """
    calls = w.decisions_per_day * (1 - w.cache_hit_rate)
    if fanout:
        tokens = calls * (w.state_tokens + w.questions_per_call * w.tokens_per_question)
        n_calls = calls
    else:
        tokens = calls * w.questions_per_call * (w.state_tokens + w.tokens_per_question)
        n_calls = calls * w.questions_per_call
    usd = tokens / 1e6 * p.input_per_mtok
    return {"calls_per_day": round(n_calls), "input_tokens_per_day": round(tokens),
            "usd_per_day": round(usd, 4), "usd_per_1k_decisions": round(usd / w.decisions_per_day * 1000, 5)}


def llm_comparison(w: Workload, llm_input_per_mtok: float, llm_output_per_mtok: float,
                   llm_output_tokens: int = 120) -> dict:
    """Same workload on a generative LLM emitting JSON (for the build-vs-buy memo)."""
    calls = w.decisions_per_day * (1 - w.cache_hit_rate)
    tin = calls * (w.state_tokens + w.questions_per_call * w.tokens_per_question + 300)  # + schema prompt
    tout = calls * llm_output_tokens
    usd = tin / 1e6 * llm_input_per_mtok + tout / 1e6 * llm_output_per_mtok
    return {"usd_per_day": round(usd, 2), "usd_per_1k_decisions": round(usd / w.decisions_per_day * 1000, 4)}
