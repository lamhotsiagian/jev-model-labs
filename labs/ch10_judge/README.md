# Lab 10 - JEV-as-a-judge: accept when confident, escalate when unsure

Method from Li, Miao, Krishnan & Padman (2026), *JEV-as-a-Judge* (arXiv:2609.26550).
All numbers here are SIMULATED; the fallback judge is a seeded coin with per-category accuracy.

## Flow
1. Build 400 synthetic pairwise items in four categories: ordinary preference, evidence-grounded factuality, derivation checking, elaborate-but-wrong answers.
2. Judge each pair in one order, then in both orders; combine with aligned averaging `p(A) = 1/2 [p1(A,B) + 1 - p1(B,A)]`.
3. Report accuracy and reversal inconsistency per category, accuracy by confidence bin `q = max(p, 1-p)`, and the AUROC of `q`.
4. Fit the escalation threshold `tau` on a 30% selection split (cascade within 2 points of the fallback).
5. Re-check `tau` on the 70% held-out split: retained accuracy, escalation rate, fee ratio.
6. Show where escalations go and which confident errors the cascade cannot catch.
7. Envelope-aware cascade: route out-of-envelope workloads (derivations, style-adversarial) straight to the fallback.

## Run
```bash
python -m labs.ch10_judge.judge_lab
pytest tests/test_judge.py
```
Swap `SimulatedStrongJudge` for a real generative judge and `JEV_BACKEND=http` to measure your own workload.
