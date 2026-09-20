"""Check WCC estimates, observation weights and kcal/mol uncertainties."""

import math

import numpy as np
import pandas as pd
import pytest

from protmutmap.bidirectional import expand_bidirectional_to_observations
from protmutmap.wcc.main import wcc_from_dataframe, wcc_multi_edge_from_dataframe
from scripts.analyze_robust_graph_estimators import run_cycle_weighted_wcc_estimators


def _simple_cycle_df(ddg_ab=1.0, ddg_bc=2.0, ddg_ac=3.0, err=0.1):
    """A 3-node triangle WT-A-B with ddG along each edge.

    With the values above the cycle closes exactly (ddg_ab + ddg_bc - ddg_ac = 0)
    so cycle closure is a no-op and the per-edge error stays at `err`.
    """
    return pd.DataFrame(
        {
            "from_mutation": ["WT", "A", "WT"],
            "to_mutation": ["A", "B", "B"],
            "calc_ddG": [ddg_ab, ddg_bc, ddg_ac],
            "calc_ddG_err": [err, err, err],
        }
    )


def test_path_dependent_errors_returned_in_kcal_per_mol():
    df = _simple_cycle_df()
    result = wcc_from_dataframe(
        df,
        ref_mol="WT",
        mol1_col="from_mutation",
        mol2_col="to_mutation",
        ddg_col="calc_ddG",
        uncertainty_col="calc_ddG_err",
    )

    assert result is not None
    nodes = result["molecules"]
    pde = dict(zip(nodes, result["path_dependent_errors"]))
    pie = dict(zip(nodes, result["path_independent_errors"]))

    # All values should be plain floats (not Decimal), allowing direct CSV/numpy use.
    for v in list(pde.values()) + list(pie.values()):
        assert isinstance(v, float)
        assert not math.isnan(v)

    # WT (the reference) has zero path-dep error.
    assert pde["WT"] == pytest.approx(0.0, abs=1e-9)

    # For any non-reference node, the path-dep error must be on the order of err
    # (kcal/mol), not err**2. Sub-kcal err on a 1-edge path => path_dep ~= err.
    err = 0.1
    assert pde["A"] == pytest.approx(err, rel=1e-3)


def test_no_decimal_leaks_into_dataframe():
    """Casting result to a DataFrame must not produce Decimal-formatted strings.

    Pre-fix the values were `decimal.Decimal(...)` and serialized as 30-digit strings
    (e.g. '14.78405833333344352818457934'), making downstream metrics misleading.
    """
    df = _simple_cycle_df()
    result = wcc_from_dataframe(
        df, ref_mol="WT",
        mol1_col="from_mutation", mol2_col="to_mutation",
        ddg_col="calc_ddG", uncertainty_col="calc_ddG_err",
    )
    s = pd.Series(result["path_dependent_errors"], index=result["molecules"])
    # pd.to_numeric must succeed without conversion warnings.
    converted = pd.to_numeric(s, errors="raise")
    assert converted.dtype == np.float64


def test_weighted_wcc_channel_differs_from_unweighted_channel():
    df = pd.DataFrame(
        {
            "from_mutation": ["WT", "A", "WT"],
            "to_mutation": ["A", "B", "B"],
            "calc_ddG": [1.0, 2.0, 5.0],
            "calc_ddG_err": [10.0, 1.0, 1.0],
        }
    )
    result = wcc_from_dataframe(
        df,
        ref_mol="WT",
        mol1_col="from_mutation",
        mol2_col="to_mutation",
        ddg_col="calc_ddG",
        uncertainty_col="calc_ddG_err",
    )

    assert result["energies"][0] != result["energies"][1]


def test_cycle_weighted_wcc_sigma_and_pseudo_var_are_equivalent():
    edges = pd.DataFrame(
        {
            "source": ["test"] * 4,
            "system": ["1ABC"] * 4,
            "from_mutation": ["WT", "A", "WT", "B"],
            "to_mutation": ["A", "AB", "B", "AB"],
            "calc_ddG": [1.0, 1.0, 1.0, 5.0],
            "calc_ddG_err": [0.1, 0.1, 0.1, 0.1],
        }
    )

    nodes, edge_weights = run_cycle_weighted_wcc_estimators(
        edges,
        pd.DataFrame(),
        bar_free_scale=1.0,
        max_alt_path_len=4,
        max_paths_per_edge=16,
        min_sigma=0.05,
        cycle_weight_alpha=1.0,
    )

    wide = nodes.pivot_table(index=["source", "system", "node"], columns="method", values="energy")
    assert not edge_weights.empty
    assert "wcc_cycle_pseudo_var" in edge_weights.columns
    assert "wcc_cycle_sigma_eff" in edge_weights.columns
    assert np.allclose(
        wide["wcc_bidir_cycle_sigma_eff_a100"],
        wide["wcc_bidir_cycle_pseudo_var_a100"],
    )


def test_wcc_multi_edge_keeps_duplicate_observations():
    df = pd.DataFrame(
        {
            "from_mutation": ["WT", "WT"],
            "to_mutation": ["A", "A"],
            "calc_ddG": [1.0, 3.0],
            "calc_ddG_err": [1.0, 1.0],
        }
    )

    result = wcc_multi_edge_from_dataframe(
        df,
        ref_mol="WT",
        mol1_col="from_mutation",
        mol2_col="to_mutation",
        ddg_col="calc_ddG",
        uncertainty_col="calc_ddG_err",
    )

    energy = dict(zip(result["molecules"], result["energies"][0]))
    assert energy["A"] == pytest.approx(2.0, abs=1e-9)


def test_wcc_multi_edge_uses_bar_error_as_weight():
    df = pd.DataFrame(
        {
            "from_mutation": ["WT", "WT"],
            "to_mutation": ["A", "A"],
            "calc_ddG": [1.0, 3.0],
            "calc_ddG_err": [0.1, 10.0],
        }
    )

    result = wcc_multi_edge_from_dataframe(
        df,
        ref_mol="WT",
        mol1_col="from_mutation",
        mol2_col="to_mutation",
        ddg_col="calc_ddG",
        uncertainty_col="calc_ddG_err",
    )

    energy = dict(zip(result["molecules"], result["energies"][0]))
    assert abs(energy["A"] - 1.0) < 0.01


def test_wcc_multi_edge_excludes_invalid_sigma_and_floors_small_sigma():
    df = pd.DataFrame(
        {
            "from_mutation": ["WT", "WT"],
            "to_mutation": ["A", "A"],
            "calc_ddG": [1.0, 3.0],
            "calc_ddG_err": [np.nan, 0.001],
        }
    )

    result = wcc_multi_edge_from_dataframe(
        df,
        ref_mol="WT",
        mol1_col="from_mutation",
        mol2_col="to_mutation",
        ddg_col="calc_ddG",
        uncertainty_col="calc_ddG_err",
        min_sigma=0.05,
    )

    energy = dict(zip(result["molecules"], result["energies"][0]))
    assert energy["A"] == pytest.approx(3.0, abs=1e-9)
    assert result["edges"]["wccme_sigma"].iloc[0] == pytest.approx(0.05, abs=1e-12)


def test_bidirectional_expansion_passes_forward_and_reverse_observations_to_wccme():
    edge_df = pd.DataFrame(
        {
            "source": ["phase2_charge"],
            "system": ["1ABC"],
            "from_mutation": ["WT"],
            "to_mutation": ["A"],
            "calc_ddG": [99.0],
            "calc_ddG_err": [9.9],
        }
    )
    bidir_df = pd.DataFrame(
        {
            "source": ["phase2_charge"],
            "system": ["1ABC"],
            "from_mutation": ["WT"],
            "to_mutation": ["A"],
            "ddG_f": [1.0],
            "ddG_f_err": [1.0],
            "ddG_r": [-3.0],
            "ddG_r_err": [1.0],
            "hysteresis": [-2.0],
            "bidir_avg": [2.0],
            "bidir_err": [0.7],
        }
    )

    obs = expand_bidirectional_to_observations(edge_df, bidir_df)

    assert len(obs) == 2
    assert set(obs["wccme_observation"]) == {"forward", "reverse"}
    assert set(zip(obs["from_mutation"], obs["to_mutation"], obs["calc_ddG"])) == {
        ("WT", "A", 1.0),
        ("A", "WT", -3.0),
    }

    result = wcc_multi_edge_from_dataframe(
        obs,
        ref_mol="WT",
        mol1_col="from_mutation",
        mol2_col="to_mutation",
        ddg_col="calc_ddG",
        uncertainty_col="calc_ddG_err",
    )
    energy = dict(zip(result["molecules"], result["energies"][0]))
    assert energy["A"] == pytest.approx(2.0, abs=1e-9)
