"""JEV-as-a-judge: aligned averaging, cascade accounting, threshold fitting."""
from jevkit import JevClient
from jevkit.judge import (PairItem, SimulatedStrongJudge, Verdict, evaluate_cascade, fee_ratio,
                          fit_threshold, judge_pair)

ITEM = PairItem("x1", "When is a duplicate charge refunded?",
                "Duplicate charges are refunded within 5 business days after support confirms them.",
                "Duplicate charges are refunded instantly with a loyalty credit.", "A", "evidence",
                "Duplicate charges are refunded automatically within 5 business days after support confirms them.")


def test_both_orders_uses_aligned_average():
    v = judge_pair(JevClient(), ITEM)
    assert v.calls == 2 and 0.5 <= v.q <= 1.0
    assert abs(v.q - max(v.p_a, 1 - v.p_a)) < 1e-12
    assert v.choice == "A"


def test_cascade_accounting():
    items = [PairItem(f"i{k}", "t", "a", "b", "A", "ordinary") for k in range(4)]
    verdicts = {"i0": Verdict("i0", "A", .95, .95, True, 10, 2), "i1": Verdict("i1", "B", .3, .7, True, 10, 2),
                "i2": Verdict("i2", "A", .6, .6, False, 10, 2), "i3": Verdict("i3", "B", .02, .98, True, 10, 2)}
    fb = {i.id: "A" for i in items}
    r = evaluate_cascade(items, verdicts, fb, tau=0.9)
    assert r["escalation_rate"] == 0.5 and r["cascade_acc"] == 0.75 and r["fallback_acc"] == 1.0
    assert fit_threshold(items, verdicts, fb, max_drop=0.0)["cascade_acc"] == 1.0
    f = fee_ratio(list(verdicts.values()), escalated=2, n=4, fallback_usd_per_1k=12.0)
    assert 0.49 < f["fee_ratio"] < 0.51


def test_simulated_fallback_is_deterministic():
    a, b = SimulatedStrongJudge(), SimulatedStrongJudge()
    assert a(ITEM) == b(ITEM)
