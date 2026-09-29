# Lab 9 - Adversarial state, governance and cost

## Flow
1. Attack suite: 150 benign tickets with appended injected instructions targeting a wrong department. Measure hijack rate with no defence.
2. Defence in depth: PII redaction, regex scan, neutralization, structured state with an `untrusted_user_input` slot, a canary Noul, and a policy downgrade that removes auto-act rights from suspicious items.
3. False-positive rate of the scanner on 300 clean tickets.
4. Cost model for 1M decisions/day: fan-out vs sequential vs a generative LLM.

## Run
```bash
python -m labs.ch09_security.security_cost_lab
```
The simulator is deliberately steerable by templated injections to model a documented Jev 1.13 weak spot; attack rates against the real API will differ. Extend `ATTACKS` with paraphrases your scanner does not match.
