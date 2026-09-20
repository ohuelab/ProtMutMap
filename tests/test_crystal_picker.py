"""Tests for protmutmap.tools.crystal_picker."""

from protmutmap.tools.crystal_picker import (
    pick_nearest_crystal,
    symmetric_diff_distance,
)

# Mimic the phase2 multi-crystal lineup
CRYSTALS = [
    "WT",
    "LB455S",
    "LB455S,RB346T,FB456L",  # KP.2
    "LB455S,FB456L,QB493E",  # KP.3
]


def test_crystal_node_picks_itself():
    """A node that *is* a crystal should pick itself (distance 0)."""
    assert pick_nearest_crystal("WT", CRYSTALS) == "WT"
    assert pick_nearest_crystal("LB455S", CRYSTALS) == "LB455S"
    assert (
        pick_nearest_crystal("LB455S,RB346T,FB456L", CRYSTALS)
        == "LB455S,RB346T,FB456L"
    )


def test_simple_intermediate_close_to_wt():
    """FB456L (single mut) has dist 1 to WT and 2 to LB455S → pick WT."""
    assert pick_nearest_crystal("FB456L", CRYSTALS) == "WT"


def test_two_mut_intermediate_closest_subset():
    # LB455S,FB456L is dist 1 from LB455S (add F456L), dist 1 from KP.2
    # (remove R346T), dist 1 from KP.3 (remove Q493E). Tie at 1; alphabetical
    # tie-break picks "LB455S".
    assert pick_nearest_crystal("LB455S,FB456L", CRYSTALS) == "LB455S"


def test_kp2_relative_intermediate():
    # RB346T,FB456L: dist=1 from KP.2 (remove LB455S), dist=2 from WT.
    assert pick_nearest_crystal("RB346T,FB456L", CRYSTALS) == "LB455S,RB346T,FB456L"


def test_kp3_relative_intermediate():
    # FB456L,QB493E: dist=1 from KP.3 (remove LB455S), dist=2 from WT.
    assert pick_nearest_crystal("FB456L,QB493E", CRYSTALS) == "LB455S,FB456L,QB493E"


def test_empty_crystals_returns_none():
    assert pick_nearest_crystal("LB455S", []) is None


def test_wt_passes_through_as_empty_set():
    """WT crystal vs another WT mutation_str → distance 0."""
    assert symmetric_diff_distance("WT", "") == 0
    assert symmetric_diff_distance("WT", "WT") == 0
