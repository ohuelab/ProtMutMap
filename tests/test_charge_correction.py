"""Tests for protmutmap.charge_correction analytical Sampson/Rocklin scheme."""


import pandas as pd
import pytest

from protmutmap.charge_correction import (
    analytical_correction_kcal,
    correction_for_edge,
    apply_corrections_to_edges_df,
    amino_acid_net_charge,
    _compute_edge_delta_q,
)


def test_neutral_box_zero_correction():
    """When ΔQ=0, the correction is identically zero."""
    out = analytical_correction_kcal(delta_q=0.0, box_length_nm=8.0, epsS=78.0)
    assert out["ddG_total_kcal"] == 0.0


def test_charged_box_nonzero_correction():
    """ΔQ=-1 in an 8 nm cubic box gives a small but non-zero correction (kcal/mol)."""
    out = analytical_correction_kcal(delta_q=-1.0, box_length_nm=8.0, epsS=78.0)
    # Order of magnitude check: |ΔΔG| should be around 0.1–1 kcal/mol for a typical box.
    # ξ_LS = -2.837297, ke ≈ 138.935 kJ/mol·nm/e^2, factor=1, L=8
    # ddG_net ≈ -0.5 * 138.935 * (-2.837297) * 1 / 8 ≈ +24.6 kJ/mol
    # ddG_usv ≈ +0.5 * 138.935 * (-2.837297) * (1 - 1/78) * 1 / 8 ≈ -24.3 kJ/mol
    # Total ≈ +0.31 kJ/mol = +0.075 kcal/mol — small after epsilon cancellation
    assert abs(out["ddG_total_kcal"]) < 1.0
    assert out["ddG_net_kcal"] > 0  # positive for ΔQ=-1
    assert out["ddG_usv_kcal"] < 0


def test_amino_acid_charges():
    assert amino_acid_net_charge("R") == 1
    assert amino_acid_net_charge("K") == 1
    assert amino_acid_net_charge("D") == -1
    assert amino_acid_net_charge("E") == -1
    assert amino_acid_net_charge("Q") == 0
    assert amino_acid_net_charge("N") == 0
    assert amino_acid_net_charge("T") == 0
    assert amino_acid_net_charge("H") == 0  # neutral histidine assumption


def test_compute_edge_delta_q_single_charge_change():
    # WT → RE99Q: R(+1) → Q(0), ΔQ = -1
    assert _compute_edge_delta_q("WT", "RE99Q") == pytest.approx(-1.0)
    # WT → NI33D: N(0) → D(-1), ΔQ = -1
    assert _compute_edge_delta_q("WT", "NI33D") == pytest.approx(-1.0)
    # WT → YH103R: Y(0) → R(+1), ΔQ = +1
    assert _compute_edge_delta_q("WT", "YH103R") == pytest.approx(1.0)
    # WT → SH105T (no charge change): ΔQ = 0
    assert _compute_edge_delta_q("WT", "SH105T") == pytest.approx(0.0)


def test_compute_edge_delta_q_multi_to_multi():
    # GE98A → GE98A,RE99Q: only RE99Q is added (R→Q: -1)
    assert _compute_edge_delta_q("GE98A", "GE98A,RE99Q") == pytest.approx(-1.0)


def test_correction_for_edge_complex_minus_partners():
    """ΔΔG_corr = c_complex - c_partner1 - c_partner2."""
    # Equal box sizes everywhere: c_complex - c_p1 - c_p2 = -c_complex (since p1=p2=complex)
    edge = correction_for_edge(
        "WT", "RE99Q",
        box_length_complex_nm=8.0,
        box_length_partner1_nm=8.0,
        box_length_partner2_nm=8.0,
    )
    assert edge.delta_q == pytest.approx(-1.0)
    # All three legs equal → ddG_corr = c - c - c = -c
    expected = -edge.correction_complex_kcal
    assert edge.correction_ddG_kcal == pytest.approx(expected, abs=1e-9)


def test_only_charge_edges_filter():
    df = pd.DataFrame(
        [
            ("WT", "SB57V", -0.81, 0.16, False),
            ("WT", "TB58D", -5.61, 0.15, True),
        ],
        columns=["from_mutation", "to_mutation", "calc_ddG", "calc_ddG_err", "has_charge_change"],
    )
    out = apply_corrections_to_edges_df(df, default_box_nm=8.0, only_charge_edges=True)
    # Non-charge edge → correction 0, calc_ddG_corrected == calc_ddG
    sb57v = out[out["to_mutation"] == "SB57V"].iloc[0]
    assert sb57v["correction_ddG_kcal"] == 0.0
    assert sb57v["calc_ddG_corrected"] == pytest.approx(-0.81)
    # Charge edge gets a correction
    tb58d = out[out["to_mutation"] == "TB58D"].iloc[0]
    assert abs(tb58d["correction_ddG_kcal"]) > 0
    assert tb58d["calc_ddG_corrected"] != pytest.approx(-5.61)
