# Lab 7 - Decision microservice and LLM integration

## Flow
1. `workflows/support_triage.yaml` binds registry + policy + question-to-action map.
2. `app.py` exposes `POST /decide/{workflow}`, `/batch`, `/healthz`, `/metrics`.
3. Each request: serialize (allow-list) -> cache lookup -> async fan-out with timeout -> retries on 429/529/timeout -> policy -> JSONL decision log.
4. Failures and the `JEV_FALLBACK=escalate` flag fail closed: every action escalates.
5. `rag_integration.py` shows the three integration points: input guardrail, passage filtering, output verification.

## Run
```bash
uvicorn labs.ch07_service.app:app --port 8080
curl -s localhost:8080/decide/support_triage -H 'content-type: application/json' \
  -d '{"record":{"plan":"pro","customer_message":"Checkout is down for all our customers!"}}'
python -m labs.ch07_service.rag_integration
docker build -f labs/ch07_service/Dockerfile -t jev-decision-service .
```
**Deliverable:** containerized service with OpenAPI docs and a RAG integration example.
