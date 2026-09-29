# Lab 11 - Capstone: support operations platform

## Flow
1. Ingest a ticket; build guarded, structured state (Ch. 4, 9).
2. One fan-out call with 15 atomic questions plus the injection canary (Ch. 5).
3. Gate `assign_queue`, `page_oncall`, `offer_refund`, `flag_churn_risk` per cost (Ch. 3, 5); suspicious items are downgraded to confirm.
4. If a refund is approved, extract the order ID via candidates (Ch. 6).
5. A stubbed LLM drafts the reply; Jev re-classifies the draft and checks for promises before it is sent (Ch. 7).
6. Every answer is logged with versions for drift and audit (Ch. 8).

## Run
```bash
python -m labs.ch11_capstone.support_ops --n 200
```
See Chapter 10 for the production-readiness checklist.
