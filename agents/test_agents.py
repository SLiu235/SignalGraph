"""
agents.test_agents — standalone unit tests for the learning-strategy core.

No external test framework required:
    python -m agents.test_agents

These exercises directly guard the IRL index-mapping (the action <-> probs index
permutation bug) and the belief-aggregation anchor:
  * _irl_mle must recover a theta whose predicted action per belief matches the
    generative model, using the correct ACTION_INDEX mapping.
  * _predict / INDEX_ACTION must map argmax -> action without permutation.
  * aggregate_belief must anchor on the most credible signal when the weighted sum
    cancels out.
"""
from __future__ import annotations

import numpy as np

from . import types as T
from .strategies import (
    ACTION_INDEX, INDEX_ACTION, aggregate_belief,
    choice_utility, logit_probs, _irl_mle,
)


def _check(name: str, cond: bool) -> None:
    if not cond:
        raise AssertionError(f"[FAIL] {name}")
    print(f"  ok - {name}")


def test_action_index_roundtrip() -> None:
    """Every action must map to its own probs index and back, without permutation."""
    for a in (T.ACTION_SUPPORT, T.ACTION_COUNTER, T.ACTION_STAY):
        idx = ACTION_INDEX[a]
        assert INDEX_ACTION[idx] == a, f"{a} -> idx {idx} -> {INDEX_ACTION[idx]}"
    _check("action<->index roundtrip", True)


def test_irl_recovers_and_predicts() -> None:
    """A truth-seeking, low-stay opponent should be recovered and predicted correctly."""
    theta = {"truth": 1.5, "contrarian": 0.0, "stay": 0.0}
    pairs = [{"belief": 1, "action": 1}, {"belief": -1, "action": -1},
             {"belief": 1, "action": 1}, {"belief": -1, "action": -1}]
    rec = _irl_mle(pairs, {"truth": [0.5, 1.0, 1.5],
                           "contrarian": [0.0, 0.5, 1.0],
                           "stay": [0.0, 0.5, 1.0]}, 0.5)
    _check("IRL recovers truth-seeking theta", rec["truth"] >= rec["contrarian"])
    _check("IRL recovers low stay", rec["stay"] <= rec["truth"])
    # predicted action per belief must match the generative model's argmax
    for b in (1, -1):
        probs = logit_probs(choice_utility(rec, b), 0.5)
        pred = INDEX_ACTION[int(np.argmax(probs))]
        _check(f"prediction for belief {b}", pred == b)


def test_predict_permutation_guard() -> None:
    """The exact regression: argmax idx0(support)=+1, idx1(counter)=-1, idx2(stay)=0."""
    theta = {"truth": 2.0, "contrarian": 0.0, "stay": 0.0}
    probs = logit_probs(choice_utility(theta, 1), 0.5)
    assert int(np.argmax(probs)) == 0
    assert INDEX_ACTION[int(np.argmax(probs))] == T.ACTION_SUPPORT
    # contrarian opponent on belief -1 -> counter
    theta_c = {"truth": 0.0, "contrarian": 2.0, "stay": 0.0}
    probs_c = logit_probs(choice_utility(theta_c, 1), 0.5)
    assert int(np.argmax(probs_c)) == 1
    assert INDEX_ACTION[int(np.argmax(probs_c))] == T.ACTION_COUNTER
    _check("predict action mapping", True)


def test_aggregate_belief_anchor() -> None:
    """When the weighted sum cancels, belief anchors on the most credible signal."""
    sigs = [
        _obs("a", 1, 0.10),   # cheap-talk support (low cred)
        _obs("b", -1, 0.10),  # cheap-talk counter (low cred)
        _obs("c", 1, 0.80),   # costly support (high cred)
    ]
    sign, conv = aggregate_belief(sigs)
    assert sign == 1 and abs(conv - 0.80) < 1e-6
    _check("aggregate_belief anchors on credible signal", True)


def _obs(sender, direction, credibility):
    return T.ObservedSignal(round=0, signal_id=sender, sender=sender, role="analyst",
                            signal_type="commentary", kind="cheap_talk", content="x",
                            direction=direction, credibility=credibility)


def main() -> None:
    print("agents.test_agents")
    test_action_index_roundtrip()
    test_irl_recovers_and_predicts()
    test_predict_permutation_guard()
    test_aggregate_belief_anchor()
    print("all tests passed")


if __name__ == "__main__":
    main()
