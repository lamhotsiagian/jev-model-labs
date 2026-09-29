# Lab 6 - Working within Jev's constraints

## Flow
1. `select_span`: regex finds `A-###` candidates with context; a Choice picks the one to refund (or `none_of_these`).
2. `hierarchical_classify`: beam search (beam=2) down a taxonomy, one call per depth.
3. `count_items`: one Noul per item, count and expected count summed in code.
4. `rank_pairs`: one Noul per (query, article) pair, all in one call, sorted in code.

## Run
```bash
python -m labs.ch06_constraints.run_helpers
```
Re-check the official jaggedness page for each new model version.
**Deliverable:** `jevkit/helpers.py`.
