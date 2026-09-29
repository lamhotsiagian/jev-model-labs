# jev-model-labs -- run every lab in book order. Default backend is the offline simulator.
PY ?= python3
.PHONY: install test labs serve clean
install:
	$(PY) -m pip install -r requirements.txt
test:
	JEV_DECISION_LOG=/tmp/jev_test_decisions.jsonl $(PY) -m pytest -q
labs:
	$(PY) -m labs.ch01_probe.probe
	$(PY) -m labs.ch02_contract.contract_demo
	$(PY) -m labs.ch03_calibration.run_calibration
	$(PY) -m labs.ch04_state.state_lab
	$(PY) -m labs.ch05_policy.run_policy
	$(PY) -m labs.ch06_constraints.run_helpers
	$(PY) -m labs.ch07_service.rag_integration
	$(PY) -m labs.ch08_ops.ops_lab
	$(PY) -m labs.ch09_security.security_cost_lab
	$(PY) -m labs.ch10_judge.judge_lab
	$(PY) -m labs.ch11_capstone.support_ops
serve:
	uvicorn labs.ch07_service.app:app --port 8080 --reload
clean:
	rm -rf labs/*/out var .pytest_cache **/__pycache__
