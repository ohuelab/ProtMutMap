"""Tests for time_ps support in protmutmap.gather_results.

Covers:
- read_bar_at_time strict match on max_time_ps + status=="ok"
- bar1.log fallback (time_ps=None) unchanged
- gather_results marks an edge incomplete (NaN) when a partner leg is
  missing/unavailable at the requested cutoff
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from protmutmap.gather_results import (
    gather_results,
    read_bar_at_time,
    read_bar1_results,
)


_BAR1_LOG = textwrap.dedent(
    """\
    Finished loading prodrun/rep0/deltae.xvg with 60003 points 20001 frames
    Performing time-split BAR (equipartition of the latter half) [kcal/mol]
    BAR SPLIT 2100.000000-2300.000000: -10.30
    BAR SPLIT 2300.000000-2500.000000: -10.50
    Final estimate [kcal/mol]
    BAR -10.45 0.10
    """
)


_BAR_TIME_SERIES_CSV = textwrap.dedent(
    """\
    max_time_ps,dG_kcal,dG_err_kcal,t_start_ps,t_end_ps,n_split_chunks,min_chunk_frames,status
    1000.0,-9.80,0.30,500.0,1000.0,10,5,ok
    2000.0,-10.10,0.20,1000.0,2000.0,10,10,ok
    4000.0,-10.35,0.12,2000.0,4000.0,10,20,ok
    8000.0,,,,,10,0,exceeds_tmax
    """
)


@pytest.fixture
def leg_dir(tmp_path: Path) -> Path:
    leg = tmp_path / "leg"
    leg.mkdir()
    (leg / "bar1.log").write_text(_BAR1_LOG)
    (leg / "bar_time_series.csv").write_text(_BAR_TIME_SERIES_CSV)
    return leg


def test_read_bar1_unchanged(leg_dir: Path):
    dg, err = read_bar1_results(leg_dir / "bar1.log")
    assert dg == pytest.approx(-10.45)
    assert err == pytest.approx(0.10)


def test_read_bar_at_time_none_falls_back_to_bar1(leg_dir: Path):
    dg, err = read_bar_at_time(leg_dir, time_ps=None)
    assert dg == pytest.approx(-10.45)
    assert err == pytest.approx(0.10)


def test_read_bar_at_time_strict_match(leg_dir: Path):
    dg, err = read_bar_at_time(leg_dir, time_ps=4000.0)
    assert dg == pytest.approx(-10.35)
    assert err == pytest.approx(0.12)


def test_read_bar_at_time_no_nearest_row_fallback(leg_dir: Path):
    # 4500 ps is closer to 4000 than to 8000 — must NOT silently round.
    with pytest.raises(FileNotFoundError):
        read_bar_at_time(leg_dir, time_ps=4500.0)


def test_read_bar_at_time_status_filter_rejects_non_ok(leg_dir: Path):
    # exceeds_tmax row exists at 8000 ps but is not status=ok.
    with pytest.raises(FileNotFoundError):
        read_bar_at_time(leg_dir, time_ps=8000.0)


def test_read_bar_at_time_missing_csv_raises(tmp_path: Path):
    empty = tmp_path / "leg_no_csv"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        read_bar_at_time(empty, time_ps=4000.0)


def test_read_bar_at_time_bar1_fallback_rejects_mismatched_tmax(tmp_path: Path):
    leg = tmp_path / "leg_short"
    leg.mkdir()
    # Fixture has tmax ~ 2500 ps (last BAR SPLIT 2300-2500).
    (leg / "bar1.log").write_text(_BAR1_LOG)
    # Request 4000 ps cutoff — bar1.log's 2500 tmax is too far away.
    with pytest.raises(FileNotFoundError):
        read_bar_at_time(leg, time_ps=4000.0)


_BAR1_LOG_4NS = textwrap.dedent(
    """\
    Finished loading prodrun/rep0/deltae.xvg with 40002 points 20001 frames
    Performing time-split BAR (equipartition of the latter half) [kcal/mol]
    BAR SPLIT 2100.000000-2300.000000: -10.30
    BAR SPLIT 2300.000000-2500.000000: -10.50
    BAR SPLIT 2500.000000-2700.000000: -10.50
    BAR SPLIT 2700.000000-2900.000000: -10.50
    BAR SPLIT 2900.000000-3100.000000: -10.50
    BAR SPLIT 3100.000000-3300.000000: -10.50
    BAR SPLIT 3300.000000-3500.000000: -10.50
    BAR SPLIT 3500.000000-3700.000000: -10.50
    BAR SPLIT 3700.000000-3900.000000: -10.50
    BAR SPLIT 3900.000000-4100.000000: -10.50
    Final estimate [kcal/mol]
    BAR -10.45 0.10
    """
)


def test_read_bar_at_time_bar1_fallback_accepts_matched_tmax(tmp_path: Path):
    leg = tmp_path / "leg_4ns"
    leg.mkdir()
    # bar1.log SPLIT ends at 4100 ps → tmax=4100, |4100-4000|=100 < 200
    (leg / "bar1.log").write_text(_BAR1_LOG_4NS)
    dg, err = read_bar_at_time(leg, time_ps=4000.0)
    assert dg == pytest.approx(-10.45)
    assert err == pytest.approx(0.10)


# --------------------------------------------------------------------------- #
# gather_results: partner-missing must yield NaN ddG
# --------------------------------------------------------------------------- #


def _make_leg(root: Path, mode: str, from_fs: str, diff_fs: str,
              *, bar1: str | None, time_series: str | None) -> Path:
    leg = root / mode / from_fs / f"wt_{diff_fs}"
    leg.mkdir(parents=True)
    if bar1 is not None:
        (leg / "bar1.log").write_text(bar1)
    if time_series is not None:
        (leg / "bar_time_series.csv").write_text(time_series)
    return leg


def test_gather_results_incomplete_partner_yields_nan(tmp_path: Path):
    """When time_ps is set and partner1's bar_time_series.csv is missing,
    the edge must be marked incomplete (calc_ddG=NaN) — NOT silently treated
    as `complex - 0 - 0`.
    """
    # Build a single edge WT → E:99K (charge change, affects chain E only)
    base = tmp_path / "mutmap_work"
    base.mkdir()

    full_csv = _BAR_TIME_SERIES_CSV
    # complex has the CSV; partner1 has bar1.log only (no time series).
    _make_leg(base, "complex", "WT", "E:99K",
              bar1=_BAR1_LOG, time_series=full_csv)
    _make_leg(base, "partner1", "WT", "E:99K",
              bar1=_BAR1_LOG, time_series=None)

    links_df = pd.DataFrame([{
        "from_mutation": "WT",
        "to_mutation": "AE99K",  # encoded so MutationList.from_string parses
    }])
    # The mutation affects chain E → partner1 leg is required; partner2 is
    # filtered out by _process_partner_mode since chains B does not match.
    partner_chains = {"partner1": ["E"], "partner2": ["B"]}

    # time_ps=None: both legs read bar1.log → ddG defined.
    out_full = gather_results(links_df, base, partner_chains, time_ps=None)
    assert not np.isnan(out_full.iloc[0]["calc_ddG"])
    # complex - partner1 - 0(partner2 not affected by chain-E mutation)
    assert out_full.iloc[0]["calc_ddG"] == pytest.approx(-10.45 - (-10.45))

    # time_ps=4000: partner1 has no bar_time_series.csv → incomplete edge.
    out_4ns = gather_results(links_df, base, partner_chains, time_ps=4000.0)
    assert np.isnan(out_4ns.iloc[0]["calc_ddG"])
    assert np.isnan(out_4ns.iloc[0]["calc_ddG_err"])


def test_gather_results_all_legs_present_at_cutoff(tmp_path: Path):
    """Sanity: when both legs have a status=ok row at the cutoff, the ddG
    uses those values."""
    base = tmp_path / "mutmap_work"
    base.mkdir()
    _make_leg(base, "complex", "WT", "E:99K",
              bar1=_BAR1_LOG, time_series=_BAR_TIME_SERIES_CSV)
    _make_leg(base, "partner1", "WT", "E:99K",
              bar1=_BAR1_LOG, time_series=_BAR_TIME_SERIES_CSV)

    links_df = pd.DataFrame([{
        "from_mutation": "WT",
        "to_mutation": "AE99K",
    }])
    partner_chains = {"partner1": ["E"], "partner2": ["B"]}

    out = gather_results(links_df, base, partner_chains, time_ps=4000.0)
    # Both legs at 4000 ps have dG=-10.35
    assert out.iloc[0]["calc_ddG"] == pytest.approx(-10.35 - (-10.35))
    assert out.iloc[0]["complex_dG"] == pytest.approx(-10.35)
    assert out.iloc[0]["partner1_dG"] == pytest.approx(-10.35)
