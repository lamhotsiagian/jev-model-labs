"""Service tests: routing, caching, fail-closed fallback."""
from fastapi.testclient import TestClient

from labs.ch07_service import app as svc

client = TestClient(svc.app)
REC = {"record": {"plan": "pro", "customer_message": "Checkout is down for all our customers!"}}


def test_decide_returns_decisions_and_probabilities():
    r = client.post("/decide/support_triage", json=REC)
    assert r.status_code == 200
    body = r.json()
    assert set(body["decisions"]) == {"assign_queue", "page_oncall", "offer_refund"}
    assert "probabilities" in body["answers"]["department"]


def test_unknown_workflow_404():
    assert client.post("/decide/nope", json=REC).status_code == 404


def test_fallback_flag_fails_closed(monkeypatch):
    monkeypatch.setenv("JEV_FALLBACK", "escalate")
    body = client.post("/decide/support_triage", json=REC).json()
    assert body["fallback"] and all(d["outcome"] == "escalate" for d in body["decisions"].values())


def test_backend_failure_fails_closed(monkeypatch):
    async def boom(*a, **k):
        raise RuntimeError("down")
    monkeypatch.setattr(svc.CLIENT, "ask", boom)
    rec = {"record": {"plan": "pro", "customer_message": "unique text to avoid cache 42"}}
    body = client.post("/decide/support_triage", json=rec).json()
    assert body["fallback"] and all(d["outcome"] == "escalate" for d in body["decisions"].values())
