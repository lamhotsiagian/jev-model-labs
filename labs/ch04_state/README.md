# Lab 4 - State serializers and question design

## Flow
A. Render one DB row, one event window (ages computed in code) and one document chunk.
B. A/B test: prose dump of the whole CRM row vs. allow-listed key-value state on 300 labelled tickets.
C. Decompose "Is this urgent?" into atomic questions (`checkout_outage`, `churn_threat`, `frustration`), combine in code, and measure accuracy, recall and run-to-run consistency over 5 meaning-preserving perturbations.
D. Load `questions.yaml` into the registry, lint it, print the fingerprint.

## Run
```bash
python -m labs.ch04_state.state_lab
```
**Deliverable:** `questions.yaml` (ID, type, criteria, owner, changelog) and `jevkit/serializers.py`.
