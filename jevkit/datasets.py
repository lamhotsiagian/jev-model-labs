"""jevkit.datasets -- reproducible synthetic support tickets with ground-truth labels.

Real calibration work needs real labelled data from your domain. For the labs
we generate tickets whose true department, outage flag, order IDs and dates
are known, with a controlled share of ambiguous tickets that mix vocabulary
from two departments, so the confidence and calibration code has genuine
errors to find.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import date, timedelta

DEPARTMENTS = {
    "billing": "Charges, invoices, refunds, payment methods",
    "technical": "Bugs, errors, outages, integrations, API problems",
    "sales": "Pricing, upgrades, plans, new accounts, quotes",
    "account": "Login, password, profile, account access, security settings",
}

_SNIPPETS = {
    "billing": ["I was charged twice for my order", "please refund the duplicate charge",
                "my invoice shows the wrong amount", "the payment method was declined",
                "I need a refund for the annual invoice", "why was my card charged again"],
    "technical": ["the API returns a 500 error", "the integration stopped syncing",
                  "checkout page shows an error and will not load", "webhooks are failing since the deploy",
                  "the dashboard crashes with a bug", "the export job times out with an error"],
    "sales": ["what is the pricing for the enterprise plan", "we want to upgrade to more seats",
              "can I get a quote for 200 users", "is there a discount on the annual plan",
              "we are evaluating plans for a new team", "how does pricing change if we upgrade"],
    "account": ["I cannot login to my account", "the password reset email never arrives",
                "please change the email on my profile", "my account is locked after login attempts",
                "how do I enable two factor security", "I lost access to my account"],
}
_OPENERS = ["Hi team,", "Hello,", "Urgent:", "Good morning,", "Hey support,", ""]
_TONE = {0: ["Thanks.", "No rush.", "Appreciate it."],
         1: ["This is getting frustrating.", "Please fix this soon.", "Still waiting on this."],
         2: ["This is unacceptable and I am very angry.", "Absolutely furious, fix it now!",
             "Worst experience, I am angry and will cancel."]}
FRUSTRATION_LEVELS = ["Calm and polite", "Frustrated or impatient", "Very angry or threatening to cancel"]


@dataclass
class Ticket:
    id: str
    text: str
    department: str
    outage: bool
    frustration: int
    order_ids: list[str] = field(default_factory=list)
    refund_order: str | None = None
    ambiguous: bool = False
    created: date | None = None


def make_tickets(n: int = 400, seed: int = 7, ambiguity: float = 0.22) -> list[Ticket]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        dept = rng.choice(list(DEPARTMENTS))
        parts = [rng.choice(_OPENERS), rng.choice(_SNIPPETS[dept])]
        amb = rng.random() < ambiguity
        if amb:  # borrow vocabulary from another department
            other = rng.choice([d for d in DEPARTMENTS if d != dept])
            parts.append("also " + rng.choice(_SNIPPETS[other]).split(" ", 2)[-1])
        outage = dept == "technical" and rng.random() < 0.35
        if outage:
            parts.append(rng.choice(["Checkout is down for all our customers.",
                                     "Multiple customers report the checkout outage.",
                                     "Checkout unavailable for several customers right now."]))
        orders = [f"A-{rng.randint(100, 999)}" for _ in range(rng.choice([0, 1, 1, 2, 3]))]
        refund = None
        if orders:
            if dept == "billing":
                refund = rng.choice(orders)
                parts.append(f"Please refund order {refund}.")
                others = [o for o in orders if o != refund]
                if others:
                    parts.append(f"Order {others[0]} was fine.")
            else:
                parts.append(f"Related order: {orders[0]}.")
        frus = rng.choices([0, 1, 2], weights=[5, 3, 2])[0]
        parts.append(rng.choice(_TONE[frus]))
        created = date(2026, 9, 1) + timedelta(days=rng.randint(0, 25))
        out.append(Ticket(f"T{i:04d}", " ".join(p for p in parts if p), dept, outage,
                          frus, orders, refund, amb, created))
    return out


# ---------------------------------------------------------------------------
# Pairwise judging items for Chapter 10 (JEV-as-a-judge)
# ---------------------------------------------------------------------------
_ORDINARY = [
    ("How do I rotate my API key safely?",
     "Create a new API key in the developer console, update your services to use the new key, confirm traffic, then revoke the old key.",
     "Thanks for reaching out. We value your feedback and our team will get back to you soon."),
    ("How can I export my invoices as CSV?",
     "Open Billing, choose Invoices, select the date range and click Export CSV to download the invoices file.",
     "Our product has many great features that customers love, including dashboards and reports."),
    ("Why are my webhooks failing to deliver?",
     "Webhooks fail when the endpoint returns errors or times out; check the delivery log, fix the endpoint, then retry the failed webhooks.",
     "Please make sure you have a stable internet connection and try again later."),
    ("How do I add seats to my plan?",
     "Go to Plan settings, choose Add seats, enter the number of seats and confirm; the plan price updates on the next invoice.",
     "Seats are an important part of every subscription and teams of all sizes use them."),
]
_EVIDENCE = [
    ("Duplicate charges are refunded automatically within 5 business days after support confirms them.",
     "When is a duplicate charge refunded?",
     "Duplicate charges are refunded within 5 business days after support confirms them.",
     "Duplicate charges are refunded instantly with an extra 20 percent loyalty credit."),
    ("Annual plans can be refunded pro rata within 30 days of purchase.",
     "Can I get a refund on an annual plan?",
     "Yes, annual plans can be refunded pro rata within 30 days of purchase.",
     "Annual plans are never refundable once the first month has started."),
    ("Old API keys expire 24 hours after a new key is created.",
     "What happens to my old API key after rotation?",
     "The old API key expires 24 hours after the new key is created.",
     "The old key keeps working forever unless you email the security team."),
    ("Webhook deliveries are retried 8 times with exponential backoff.",
     "How many times are webhooks retried?",
     "Webhook deliveries are retried 8 times with exponential backoff.",
     "Webhooks are retried once, after exactly one hour, and then dropped."),
]


def _derivation(rng: random.Random):
    price, qty = rng.randint(2, 9), rng.randint(3, 9)
    paid = (price * qty // 10 + 2) * 10
    total, change = price * qty, paid - price * qty
    wrong_total = total + rng.choice([-3, -2, 2, 3])
    task = (f"Pens cost {price} dollars each. Sam buys {qty} pens and pays with {paid} dollars. "
            f"How much change does Sam get? Show the working.")
    good = f"Cost is {qty} x {price} = {total} dollars, so change is {paid} - {total} = {change} dollars."
    bad = f"Cost is {qty} x {price} = {wrong_total} dollars, so change is {paid} - {wrong_total} = {paid - wrong_total} dollars."
    return task, good, bad


_ELABORATE = [
    ("How long does a refund for a duplicate charge take?",
     "About 5 business days.",
     "Refund timing for a duplicate charge depends on how the refund for the charge is processed: "
     "every duplicate charge refund takes 30 business days because each refund for a duplicate "
     "charge must be reviewed by the billing refund team before the charge is refunded."),
    ("Can webhooks be retried after they fail?",
     "Yes, failed webhooks are retried automatically.",
     "Webhooks that fail cannot be retried: when webhooks fail the failed webhooks are permanently "
     "dropped, so retried webhooks never happen and you must recreate every failed webhook manually."),
    ("Do annual plans support refunds?",
     "Yes, within 30 days, pro rata.",
     "Annual plans and annual plan refunds work differently from monthly plans: annual plans never "
     "support refunds, and an annual plan refund request for annual plans is always declined."),
    # milder variants: the wrong answer is longer but not keyword-stuffed
    ("How long does a refund for a duplicate charge take?",
     "A duplicate charge refund takes about 5 business days after support confirms it.",
     "In our experience these things can take a while, and it is usually best to wait a month or so "
     "before following up, since banks and processors have their own timelines."),
    ("Can webhooks be retried after they fail?",
     "Yes, failed webhooks are retried automatically with backoff.",
     "It depends on many factors in your infrastructure, but generally speaking once something has "
     "failed you should assume it is gone and plan to rebuild it from your own records."),
    ("Do annual plans support refunds?",
     "Yes, annual plans support pro rata refunds within 30 days.",
     "Most subscription businesses avoid giving money back on long commitments, and customers "
     "generally understand that a year-long purchase is final once it has been made."),
]


@dataclass
class _Pair:
    pass


def make_judge_pairs(n: int = 400, seed: int = 11):
    """Balanced synthetic pairwise items across four categories with known labels.

    ordinary        clear preference between a relevant and an irrelevant answer
    evidence        evidence-grounded factuality: one answer contradicts the passage
    derivation      same wording, one arithmetic slip: requires checking a derivation
    elaborate_wrong short correct answer vs. long, keyword-rich, wrong answer
    """
    from .judge import PairItem
    rng = random.Random(seed)
    cats = ["ordinary", "evidence", "derivation", "elaborate_wrong"]
    out = []
    for i in range(n):
        cat = cats[i % 4]
        evidence = None
        if cat == "ordinary":
            task, good, bad = rng.choice(_ORDINARY)
        elif cat == "evidence":
            evidence, task, good, bad = rng.choice(_EVIDENCE)
        elif cat == "derivation":
            task, good, bad = _derivation(rng)
        else:
            task, good, bad = rng.choice(_ELABORATE)
        tag = f" (ref {i})"                      # makes each item unique for caching/noise
        if rng.random() < 0.5:
            out.append(PairItem(f"J{i:04d}", task + tag, good, bad, "A", cat, evidence))
        else:
            out.append(PairItem(f"J{i:04d}", task + tag, bad, good, "B", cat, evidence))
    return out
