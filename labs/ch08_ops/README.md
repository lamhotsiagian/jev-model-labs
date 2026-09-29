# Lab 8 - Production operations

## Flow
1. Golden-set gate: pinned `jev-1.13.0` vs. candidate `jev-1.14.0` (simulated) on 300 labelled items; block on >1 point regression and review right-to-wrong flips.
2. Log a baseline week and a week with a new enterprise segment to JSONL.
3. `drift_report`: PSI and escalation-rate change per question.
4. Audit loop: sampled auto-acts with human labels; disagreements return to the labelled set.
5. Weekly calibration report (`out/weekly_report.json`).

Also here: `upgrade_runbook.md` and `grafana_dashboard.json`.

## Run
```bash
python -m labs.ch08_ops.ops_lab
```
**Deliverable:** dashboard, upgrade runbook, weekly calibration report job.
