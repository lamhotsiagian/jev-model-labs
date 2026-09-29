# Runbook: upgrading the pinned Jev version

**Owner:** decision-platform on-call · **Trigger:** TypeSafe announces a new version, or `jev-preview` shows a gain on the golden set.

1. **Read the release notes and the jaggedness page** for the new version. List every weak spot that changed.
2. **Freeze inputs.** Record the golden set commit, registry fingerprint and policy version.
3. **Shadow run.** Evaluate the candidate on the golden set with the *current* thresholds:
   `python -m labs.ch08_ops.ops_lab` (gate: accuracy regression <= 1 point, right-to-wrong flips reviewed by hand).
4. **Re-calibrate.** Run `labs/ch03_calibration` against the candidate. Thresholds are tuned per version; never carry them over blindly.
5. **Canary.** Route 5% of traffic to the candidate for 48 h. Compare escalation rate, PSI and audit disagreement per question.
6. **Promote.** Change the pinned version in config (one PR: version + any threshold changes + calibration report attached).
7. **Rollback.** Revert the config PR. Because every log line carries the concrete model version, you can attribute any incident window to one version.

**Never:** point production at `jev-latest`; change a threshold and a model version in separate unreviewed steps; delete the previous version's calibration report.
