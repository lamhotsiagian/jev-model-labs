# Lab 1 - Black-box probe harness

**Goal.** Verify three properties yourself instead of trusting a launch post.

| Claim | Probe | Pass condition |
|---|---|---|
| P1 parallel sampler | median latency at 1, 5, 10, 25, 50 questions | 50 questions cost well under 2x one question |
| P2 isolation | question A alone vs. A with 20 distractors | probabilities equal within rounding (0.01) |
| P3 state cost | 1x..16x the same state, 10 questions | latency grows with input tokens |

## Flow
1. `data/sample_state.txt` is a 40-ticket queue export (about 1.3k tokens).
2. `p1_question_count` sends N trivial Noul questions; each point is the median of 5 runs with a varied state so no cache can help.
3. `p2_isolation` compares a Noul and a full Choice distribution alone vs. in a crowd.
4. `p3_state_length` repeats the state 1..16 times.
5. Results go to `out/probe.json`, `out/latency.png`, `out/report.md`.

## Run
```bash
python -m labs.ch01_probe.probe                      # SIMULATED
JEV_BACKEND=http python -m labs.ch01_probe.probe     # your region, real numbers
```
**Deliverable:** the `probe/` module and a one-page report with latency curves from your own region.
