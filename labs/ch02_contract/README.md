# Lab 2 - A typed domain wrapper

## Flow
1. Build a request with all three primitives (`ChoiceQ`, `NoulQ`, `ScoreQ`) and print the exact wire body.
2. Wrap the backend in `RecordingBackend` so the raw response is saved to `data/fixtures/<hash>.json`.
3. `parse_response` converts it to typed answers, renormalizes 2-dp probabilities and checks that every answer type and Choice label matches what was asked.
4. `check_score_monotonic` lints the rubric order (a reversed rubric silently inverts every threshold).
5. `to_view` maps answers onto `TriageView`, the only object the rest of the app sees.

## Run
```bash
python -m labs.ch02_contract.contract_demo
JEV_BACKEND=replay pytest tests/test_contract.py   # tests never hit the API
```
**Deliverable:** `jevkit/contract.py` and `tests/test_contract.py` on recorded fixtures.
