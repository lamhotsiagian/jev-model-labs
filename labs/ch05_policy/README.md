# Lab 5 - The decision layer as code

## Flow
1. Speculative fan-out: all 12 registry questions in one call vs. 12 sequential calls (tokens, latency).
2. Per-action gating from `policy.yaml` (`assign_queue` 0.70, `auto_close_ticket` 0.95, `page_oncall` on P(yes)).
3. Composite priority with weights from YAML.
4. Intent routing: a Choice selects the handler; low confidence goes to a human.
5. Break-even thresholds derived from the cost fields.

## Run
```bash
python -m labs.ch05_policy.run_policy
pytest tests/test_toolkit.py -k policy
```
**Deliverable:** `jevkit/policy.py` with thresholds loaded from YAML, unit tested with synthetic answers.
