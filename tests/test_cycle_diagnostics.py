"""Test cycle scores, tied-edge reporting and cycle-size limits."""


import pandas as pd
import pytest

from protmutmap.cycle_diagnostics import compute_edge_cycle_scores


def _mlc_like_network() -> pd.DataFrame:
    """Two four-edge cycles sharing an edge, with closure errors 5.27 and 0.42."""
    rows = [
        # cycle 1 (high hysteresis): WT-SB57V-SB57V_TB58D-TB58D-WT, |delta|=5.27
        ("WT", "SB57V", -0.81, 0.163, False),
        ("WT", "TB58D", -5.61, 0.153, True),
        ("SB57V", "SB57V_TB58D", -0.35, 0.156, True),
        ("TB58D", "SB57V_TB58D", -0.82, 0.156, False),
        # cycle 2 (low hysteresis): TB58D-NA92A_TB58D-NA92A_SB57V_TB58D-SB57V_TB58D-TB58D, |delta|=0.42
        ("TB58D", "NA92A_TB58D", 0.05, 0.192, False),
        ("NA92A_TB58D", "NA92A_SB57V_TB58D", -0.68, 0.256, False),
        ("SB57V_TB58D", "NA92A_SB57V_TB58D", -0.23, 0.192, False),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "from_mutation", "to_mutation", "calc_ddG", "calc_ddG_err", "has_charge_change",
        ],
    )


def test_high_hysteresis_cycle_dominates_4_cycle_only():
    """With max_cycle_size=4 (only the two primary 4-cycles), edges from the high-
    hysteresis cycle should score 5.27/4 ≈ 1.32, vs 0.42/4 ≈ 0.105 for low-cycle edges."""
    df = _mlc_like_network()
    out = compute_edge_cycle_scores(df, max_cycle_size=4)

    score = {(r["from_mutation"], r["to_mutation"]): r["edge_cycle_score"] for _, r in out.iterrows()}

    # Edges in high-hysteresis cycle only (not the shared TB58D-SB57V_TB58D)
    cycle1_only = [
        ("WT", "SB57V"), ("WT", "TB58D"), ("SB57V", "SB57V_TB58D"),
    ]
    cycle2_only = [
        ("TB58D", "NA92A_TB58D"),
        ("NA92A_TB58D", "NA92A_SB57V_TB58D"),
        ("SB57V_TB58D", "NA92A_SB57V_TB58D"),
    ]

    # Each cycle-1-only edge sees only delta1=5.27 over N=4 -> 1.32
    for e in cycle1_only:
        assert score[e] == pytest.approx(5.27 / 4, abs=0.02), f"{e} cycle1-only score off"
    # Each cycle-2-only edge sees only delta2=0.42 over N=4 -> 0.105
    for e in cycle2_only:
        assert score[e] == pytest.approx(0.42 / 4, abs=0.02), f"{e} cycle2-only score off"
    # Shared edge sees both, median(1.32, 0.105) = 0.7125
    shared = ("TB58D", "SB57V_TB58D")
    assert score[shared] == pytest.approx((1.32 + 0.105) / 2, abs=0.05)


def test_identifiability_three_edges_tied_4_cycle():
    """Report ties when a cycle cannot distinguish which edge caused its error."""
    df = _mlc_like_network()
    out = compute_edge_cycle_scores(df, max_cycle_size=4)
    score = {(r["from_mutation"], r["to_mutation"]): (r["edge_cycle_score"], r["cycle_score_tied_n"])
             for _, r in out.iterrows()}

    # The three cycle-1-only edges should share the same score
    cycle1_only_score = score[("WT", "TB58D")][0]
    same_score_edges = [
        ("WT", "TB58D"),
        ("WT", "SB57V"),
        ("SB57V", "SB57V_TB58D"),
    ]
    for e in same_score_edges:
        assert score[e][0] == pytest.approx(cycle1_only_score, abs=1e-6), (
            f"{e} should be tied with WT->TB58D"
        )
    # tied_n for each tied edge >= 2 (n_other_edges_with_same_score)
    for e in same_score_edges:
        assert score[e][1] >= 2, f"{e} cycle_score_tied_n={score[e][1]} should be >= 2"


def test_default_includes_6_cycle():
    """With the default max_cycle_size=6, the wrap-around 6-cycle is also counted.
    This verifies n_cycles increases for cycle-1/2-only edges accordingly."""
    df = _mlc_like_network()
    out = compute_edge_cycle_scores(df)  # max_cycle_size=6 default
    n_cyc = {(r["from_mutation"], r["to_mutation"]): r["n_cycles"] for _, r in out.iterrows()}

    # The shared edge TB58D-SB57V_TB58D is in BOTH 4-cycles. The 6-cycle goes around
    # the outer perimeter (WT-SB57V-...-TB58D-WT) and skips the shared diagonal —
    # so the shared edge stays at n_cycles=2.
    assert n_cyc[("TB58D", "SB57V_TB58D")] == 2
    # WT->TB58D is in 4-cycle (cycle 1) AND 6-cycle wrap-around → 2 cycles
    assert n_cyc[("WT", "TB58D")] == 2
    # Same for the other cycle-1-only edges
    assert n_cyc[("WT", "SB57V")] == 2
    assert n_cyc[("SB57V", "SB57V_TB58D")] == 2


def test_no_cycles_returns_zeros():
    """Tree-only graph: no cycles, all edges get score=0, n_cycles=0."""
    df = pd.DataFrame(
        [
            ("WT", "A", -1.0, 0.1, False),
            ("A", "B", -2.0, 0.1, False),
            ("B", "C", -3.0, 0.1, False),
        ],
        columns=["from_mutation", "to_mutation", "calc_ddG", "calc_ddG_err", "has_charge_change"],
    )
    out = compute_edge_cycle_scores(df)
    assert all(out["n_cycles"] == 0)
    assert all(out["edge_cycle_score"] == 0.0)
