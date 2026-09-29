"""Chapter 1 lab -- black-box probe harness.

Verifies three disclosed/claimed properties of Jev yourself, from your region:

  P1  latency is nearly flat as the number of questions grows (parallel sampler)
  P2  questions are isolated (asking A alone == asking A among 20 others)
  P3  latency scales with state length

Run:
    python -m labs.ch01_probe.probe                 # simulator (SIMULATED numbers)
    JEV_BACKEND=http python -m labs.ch01_probe.probe  # real API, real numbers
Writes labs/ch01_probe/out/{probe.json, latency.png, report.md}.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

from jevkit import ChoiceQ, JevClient, NoulQ

OUT = Path(__file__).parent / "out"
STATE = Path("data/sample_state.txt").read_text()
REPEATS = 5


def median_latency(client: JevClient, state: str, qs: dict) -> tuple[float, dict]:
    lat, usage = [], {}
    for r in range(REPEATS):
        # vary state trivially so no cache (ours or the vendor's) can help
        res = client.ask(state + f"\n<!-- probe run {r} -->", qs)
        lat.append(res.latency_ms)
        usage = {"input_tokens": res.input_tokens, "output_tokens": res.output_tokens}
    return statistics.median(lat), usage


def p1_question_count(client: JevClient) -> list[dict]:
    rows = []
    for n in [1, 5, 10, 25, 50]:
        qs = {f"q{i}": NoulQ(instructions=f"The text discusses topic #{i}") for i in range(n)}
        ms, usage = median_latency(client, STATE, qs)
        rows.append({"n_questions": n, "median_ms": round(ms, 1), **usage})
        print(f"P1 n={n:>3}  median={ms:7.1f} ms  usage={usage}")
    return rows


def p2_isolation(client: JevClient) -> dict:
    target = {"refund": NoulQ("Does any customer ask for money back?")}
    distractors = {f"d{i}": NoulQ(f"Does the text mention the color number {i}?") for i in range(20)}
    alone = client.ask(STATE, target)["refund"].p_true
    crowd = client.ask(STATE, {**distractors, **target})["refund"].p_true
    # also a Choice, whose full distribution must match
    ch = {"team": ChoiceQ("Which team handles most of these tickets?",
                          {"billing": "Charges and refunds", "technical": "Bugs and outages",
                           "sales": "Pricing and plans", "account": "Login and access"})}
    c_alone = client.ask(STATE, ch)["team"].raw_probabilities
    c_crowd = client.ask(STATE, {**distractors, **ch})["team"].raw_probabilities
    max_diff = max(abs(c_alone[k] - c_crowd[k]) for k in c_alone)
    res = {"noul_alone": alone, "noul_with_20": crowd, "noul_diff": round(abs(alone - crowd), 3),
           "choice_max_abs_diff": round(max_diff, 3),
           "isolated": abs(alone - crowd) <= 0.01 and max_diff <= 0.01}
    print("P2", res)
    return res


def p3_state_length(client: JevClient) -> list[dict]:
    rows, qs = [], {f"q{i}": NoulQ(f"The text discusses topic #{i}") for i in range(10)}
    for mult in [1, 2, 4, 8, 16]:
        ms, usage = median_latency(client, STATE * mult, qs)
        rows.append({"state_copies": mult, "input_tokens": usage["input_tokens"], "median_ms": round(ms, 1)})
        print(f"P3 x{mult:<3} tokens={usage['input_tokens']:>6}  median={ms:7.1f} ms")
    return rows


def main() -> None:
    OUT.mkdir(exist_ok=True)
    client = JevClient()
    tag = "SIMULATED" if client.is_simulated else "MEASURED"
    data = {"backend": tag, "p1": p1_question_count(client), "p2": p2_isolation(client),
            "p3": p3_state_length(client)}
    (OUT / "probe.json").write_text(json.dumps(data, indent=2))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
        ax[0].plot([r["n_questions"] for r in data["p1"]], [r["median_ms"] for r in data["p1"]], "o-")
        ax[0].set(xlabel="questions per call", ylabel="median latency (ms)", title="P1: fan-out", ylim=(0, None))
        ax[1].plot([r["input_tokens"] for r in data["p3"]], [r["median_ms"] for r in data["p3"]], "s-", color="C2")
        ax[1].set(xlabel="input tokens", ylabel="median latency (ms)", title="P3: state length")
        fig.suptitle(f"Jev probe ({tag})")
        fig.tight_layout()
        fig.savefig(OUT / "latency.png", dpi=160)
    except ImportError:
        pass
    p1, p3 = data["p1"], data["p3"]
    report = [f"# Jev probe report ({tag})", "",
              f"* P1 latency 1 -> 50 questions: {p1[0]['median_ms']} -> {p1[-1]['median_ms']} ms "
              f"({p1[-1]['median_ms'] / p1[0]['median_ms']:.2f}x for 50x the questions)",
              f"* P2 isolation holds: {data['p2']['isolated']} (max abs diff {data['p2']['choice_max_abs_diff']})",
              f"* P3 latency {p3[0]['input_tokens']} -> {p3[-1]['input_tokens']} tokens: "
              f"{p3[0]['median_ms']} -> {p3[-1]['median_ms']} ms"]
    (OUT / "report.md").write_text("\n".join(report) + "\n")
    print("\n".join(report))


if __name__ == "__main__":
    main()
