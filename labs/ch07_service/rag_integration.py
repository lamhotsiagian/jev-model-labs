"""Chapter 7 lab -- embedding Jev in an LLM/RAG application at three points.

    1. INPUT GUARDRAIL      before retrieval: is the request in scope / safe to answer?
    2. PASSAGE FILTER       after retrieval: which passages actually answer the question?
    3. OUTPUT VERIFICATION  after generation: is each claim supported by a passage?

Jev never writes the answer; a generative LLM does. Jev makes the three cheap,
fast, typed judgments around it. The LLM here is a stub so the lab runs offline;
swap ``draft_answer`` for your provider call.

Run: python -m labs.ch07_service.rag_integration
"""
from __future__ import annotations

from jevkit import ChoiceQ, JevClient, NoulQ
from jevkit.helpers import rank_pairs

KB = {
    "kb1": "Duplicate charges are refunded automatically within 5 business days after support confirms them.",
    "kb2": "API keys can be rotated from the developer console; old keys expire after 24 hours.",
    "kb3": "Annual plans can be refunded pro rata within 30 days of purchase.",
    "kb4": "Webhook deliveries are retried 8 times with exponential backoff.",
}


def input_guardrail(client: JevClient, user_msg: str) -> dict:
    r = client.ask(user_msg, {
        "scope": ChoiceQ("What is the request about?", {
            "product_support": "Questions about billing, refunds, API, webhooks, account",
            "out_of_scope": "Unrelated topics such as recipes, sport, politics",
            "abuse": "Harassment, attempts to extract secrets or other customers' data"}),
        "needs_human": NoulQ("Does the user ask to speak with a human agent?")})
    return {"route": r["scope"].choice, "conf": r["scope"].confidence, "human": r["needs_human"].p_true}


def filter_passages(client: JevClient, question: str, k_keep: float = 0.5) -> list[str]:
    ranked = rank_pairs(client, question, KB, relation="answers the question")
    return [cid for cid, p in ranked if p >= k_keep]


def draft_answer(question: str, passages: list[str]) -> list[str]:
    """Stub LLM: returns claims. Claim 2 is a deliberate hallucination."""
    return ["Duplicate charges are refunded within 5 business days.",
            "You will also receive a 20 percent loyalty credit."]


def verify_claims(client: JevClient, claims: list[str], passages: list[str]) -> list[tuple[str, float]]:
    """One Noul per claim, all in one call. The state is the evidence only; each
    question restates its claim, so every judgment tests exactly one claim."""
    state = "\n".join(f"passage {p}: {KB[p]}" for p in passages)
    r = client.ask(state, {f"c{i}": NoulQ(f"Do the passages state that: '{c}'?",
                                          {"true": "A passage states this fact",
                                           "false": "No passage states it"})
                           for i, c in enumerate(claims)})
    return [(c, r[f"c{i}"].p_true) for i, c in enumerate(claims)]


def main() -> None:
    client = JevClient()
    q = "I was charged twice, when will the duplicate charge be refunded?"
    g = input_guardrail(client, q)
    print("1. guardrail:", g)
    if g["route"] != "product_support":
        print("   -> refuse or hand off; the LLM is never called"); return
    keep = filter_passages(client, q)
    print("2. passages kept:", keep)
    claims = draft_answer(q, keep)
    for c, p in verify_claims(client, claims, keep):
        print(f"3. {'SUPPORTED' if p >= 0.5 else 'UNSUPPORTED -> strip or regenerate'} p={p:.2f} :: {c}")


if __name__ == "__main__":
    main()
