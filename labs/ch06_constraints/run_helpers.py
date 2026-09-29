"""Chapter 6 lab -- working within Jev's constraints.

Flow:
  1. select_span: extract the order the customer wants refunded (not just any order ID);
  2. hierarchical_classify: a 3-level product taxonomy (flattened > 255 leaves in real life);
  3. count_items: count semantic matches with one Noul per item, summed in code;
  4. rank_pairs: rerank knowledge-base articles for a query in one call.

Run: python -m labs.ch06_constraints.run_helpers
"""
from __future__ import annotations

import re

from jevkit import JevClient
from jevkit.datasets import make_tickets
from jevkit.helpers import count_items, hierarchical_classify, rank_pairs, select_span

ORDER_RX = re.compile(r"\bA-\d{3}\b")

TAXONOMY = {
    "payments": {"_desc": "money moving: charges, refunds, invoices",
                 "refunds": {"duplicate_charge": "charged twice for one order",
                             "partial_refund": "refund part of an order"},
                 "invoicing": {"wrong_amount": "invoice shows the wrong amount",
                               "tax_question": "VAT or sales tax on invoice"}},
    "platform": {"_desc": "software problems: API, webhooks, errors, outages",
                 "api": {"http_500": "API returns 500 error", "auth_401": "API key rejected"},
                 "webhooks": {"delivery_failure": "webhooks failing to deliver",
                              "signature_mismatch": "webhook signature invalid"}},
    "identity": {"_desc": "login, password, account access",
                 "login": {"locked_out": "account locked after login attempts",
                           "reset_email": "password reset email never arrives"}},
}


def main() -> None:
    client = JevClient()
    print("== 1. extraction by candidates ==")
    hits = total = 0
    for t in [t for t in make_tickets(300, seed=4) if t.refund_order][:40]:
        r = select_span(client, t.text, ORDER_RX, "the order the customer wants refunded")
        total += 1
        hits += r.value == t.refund_order
    print(f"refund order extracted correctly: {hits}/{total}")
    t = next(t for t in make_tickets(300, seed=4) if t.refund_order and len(t.order_ids) > 1)
    print("example:", t.text, "\n  ->", select_span(client, t.text, ORDER_RX, "the order the customer wants refunded"))

    print("\n== 2. hierarchical classification with beam search ==")
    text = "Our webhooks stopped delivering since yesterday, deliveries failing with timeouts."
    for path, p in hierarchical_classify(client, text, TAXONOMY, beam=2)[:4]:
        print(f"  {p:.3f}  {path}")

    print("\n== 3. counting: one Noul per item ==")
    items = ["Refund me for order A-101", "Love the new dashboard", "I want my money back",
             "How do I export CSV?", "Please reverse the charge on my card", "Great support team"]
    print(count_items(client, items, "the customer asks for money back",
                      criteria={"true": "asks for a refund, reversal, or money back",
                                "false": "praise, questions or anything else"}))

    print("\n== 4. pairwise reranking in one call ==")
    kb = {"kb1": "How to request a refund for a duplicate charge",
          "kb2": "Rotating your API key safely",
          "kb3": "Understanding invoice line items and refunds",
          "kb4": "Configuring webhook retries"}
    for cid, p in rank_pairs(client, "I was charged twice and need a refund", kb):
        print(f"  {p:.2f} {cid} {kb[cid]}")


if __name__ == "__main__":
    main()
