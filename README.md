# jev-model-labs

Companion code for **Engineering Decision Systems with JEV** (AI Engineering Insider).
A production-style toolkit (`jevkit`) plus eleven hands-on labs, one per chapter,
for building decision systems on TypeSafe's Jev System One model.

> **Runs offline by default.** Jev is a hosted, early-access API. Every lab runs
> against `jevkit.simulator.JevSimulator`, a deterministic stand-in that returns
> the exact `/v1/systemone` wire format and reproduces documented behaviours
> (fan-out, isolation, 2-dp rounding, concrete version IDs, injection
> susceptibility). Its *judgment* is a keyword heuristic, **not Jev**. Every
> number printed under the simulator is labelled `SIMULATED`. Set
> `JEV_BACKEND=http` and `TYPESAFE_API_KEY` to measure the real model.

<img width="1241" height="1754" alt="jev-prev_page-0001" src="https://github.com/user-attachments/assets/d6859a72-8b61-4b8c-a2e7-8f7a31ef0b7a" />

preview: https://drive.google.com/file/d/1NsGSt--MJQwDnU-nXA7IG-VVtn6Rev_G/view

book link: https://shop.beacons.ai/aiengineeringinsider/98bafdb3-a49d-4d7e-80d1-1255f03438af
## Quick start

```bash
python -m pip install -r requirements.txt
make test          # unit tests, no network
make labs          # run all eleven labs in book order
make serve         # decision microservice on :8080 (OpenAPI at /docs)
```

## Layout

| Path | Chapter | Purpose |
|---|---|---|
| `jevkit/contract.py` | 2 | Typed questions/answers, normalization, validation |
| `jevkit/client.py` | 1, 2, 7 | Sync/async client; `sim`, `http`, `replay` backends; retries |
| `jevkit/simulator.py` | all | Offline stand-in for the API |
| `jevkit/confidence.py` | 3 | Peakedness (community-derived), top-2 margin, entropy |
| `jevkit/calibration.py` | 3 | ECE, Brier, reliability bins, threshold sweep |
| `jevkit/registry.py` | 4 | Versioned question registry with lint and fingerprints |
| `jevkit/serializers.py` | 4 | Row, event-window and document-chunk state serializers |
| `jevkit/policy.py` | 5 | Per-action thresholds, composites, intent routing |
| `jevkit/helpers.py` | 6 | `select_span`, `hierarchical_classify`, `count_items`, `rank_pairs` |
| `jevkit/cache.py` | 7 | Cache keyed on model version + state hash + question versions |
| `jevkit/telemetry.py` | 8 | Decision log, PSI drift, golden-set upgrade gate |
| `jevkit/security.py` | 9 | Injection scan, PII redaction, canary question |
| `jevkit/cost.py` | 9 | Unit economics: fan-out vs sequential vs LLM |
| `jevkit/judge.py` | 10 | JEV-as-a-judge: both-order pairwise judging, confidence cascade |
| `labs/chNN_*` | 1-11 | One lab per chapter, each with its own README |

## Backends

| `JEV_BACKEND` | What it does |
|---|---|
| `sim` (default) | Offline simulator. Add `JEV_SIM_SLEEP=1` to really sleep the modelled latency. |
| `http` | `POST https://api.typesafe.ai/v1/systemone` with `Authorization: Bearer $TYPESAFE_API_KEY`; exponential backoff on 429/529. |
| `replay` | Serves recorded fixtures from `JEV_FIXTURES` (default `data/fixtures`). |

## Lab flow

1. `ch01_probe` - verify flat fan-out latency, isolation and state-length scaling.
2. `ch02_contract` - typed wrapper, fixtures, normalization, rubric lint.
3. `ch03_calibration` - compare confidence measures, ECE/Brier, pick a threshold.
4. `ch04_state` - serializers, prose vs key-value A/B, atomic vs holistic questions.
5. `ch05_policy` - fan-out, per-action gating, composites, intent routing.
6. `ch06_constraints` - extraction, >255 classes, counting, pairwise ranking.
7. `ch07_service` - FastAPI decision service + RAG integration points.
8. `ch08_ops` - golden-set gate, drift, audit loop, weekly calibration report.
9. `ch09_security` - injection attack suite, defence in depth, cost model.
10. `ch10_judge` - JEV-as-a-judge: both-order pairwise judging, confidence cascade with escalation.
11. `ch11_capstone` - support operations platform using every module.

Author: AI Engineering Insider · http://aiengineeringinsider.com ·
Newsletter: https://aiengineeringinsider.substack.com/subscribe
# jev-model-labs
