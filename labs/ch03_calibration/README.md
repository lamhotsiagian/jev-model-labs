# Lab 3 - Confidence toolkit and calibration

## Flow
1. Generate 400 labelled tickets (`jevkit.datasets.make_tickets`); in production use 300-500 of your own labelled items.
2. Ask the department Choice for each and keep the full distribution.
3. `verify_against_returned` checks the community peakedness formula against the returned `confidence`.
4. `compare_measures` ranks peakedness, top-2 margin and entropy by AUROC for predicting correctness.
5. Compute ECE and Brier, draw `out/reliability.png`.
6. `threshold_sweep` + `recommend_threshold --error-budget 0.05` produce `out/tradeoff.png` and a threshold.

## Run
```bash
python -m labs.ch03_calibration.run_calibration --error-budget 0.05
```
**Deliverable:** `jevkit/calibration.py` plus a threshold recommendation derived from labelled data.
