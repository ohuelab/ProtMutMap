import pandas as pd
import pytest

from protmutmap.robust_graph import cycle_residual_edge_scores, fit_node_potentials
from scripts.analyze_robust_graph_estimators import expand_bidirectional_edges


def test_fit_node_potentials_recovers_consistent_graph():
    edges = pd.DataFrame(
        [
            {"from_mutation": "WT", "to_mutation": "A", "calc_ddG": 1.0, "calc_ddG_err": 0.1},
            {"from_mutation": "A", "to_mutation": "AB", "calc_ddG": 2.0, "calc_ddG_err": 0.1},
            {"from_mutation": "WT", "to_mutation": "B", "calc_ddG": -1.0, "calc_ddG_err": 0.1},
            {"from_mutation": "B", "to_mutation": "AB", "calc_ddG": 4.0, "calc_ddG_err": 0.1},
        ]
    )

    fit = fit_node_potentials(edges, method="huber")
    energy = dict(zip(fit.nodes["node"], fit.nodes["energy"]))

    assert fit.converged
    assert energy["WT"] == pytest.approx(0.0)
    assert energy["A"] == pytest.approx(1.0)
    assert energy["B"] == pytest.approx(-1.0)
    assert energy["AB"] == pytest.approx(3.0)
    assert fit.edges["node_fit_residual"].abs().max() < 1e-8


def test_cycle_residual_scores_require_alternative_paths():
    edges = pd.DataFrame(
        [
            {"from_mutation": "WT", "to_mutation": "A", "calc_ddG": 1.0, "calc_ddG_err": 0.1},
            {"from_mutation": "A", "to_mutation": "AB", "calc_ddG": 1.0, "calc_ddG_err": 0.1},
            {"from_mutation": "WT", "to_mutation": "B", "calc_ddG": 1.0, "calc_ddG_err": 0.1},
            {"from_mutation": "B", "to_mutation": "AB", "calc_ddG": 5.0, "calc_ddG_err": 0.1},
            {"from_mutation": "LONE", "to_mutation": "X", "calc_ddG": 2.0, "calc_ddG_err": 0.1},
        ]
    )

    scores = cycle_residual_edge_scores(edges, max_path_len=4)
    keyed = scores.set_index(["from_mutation", "to_mutation"])

    assert keyed.loc[("LONE", "X"), "cycle_alt_n_paths"] == 0
    assert keyed.loc[("LONE", "X"), "cycle_weight_factor"] == pytest.approx(1.0)
    assert keyed.loc[("B", "AB"), "cycle_alt_n_paths"] >= 1
    assert keyed.loc[("B", "AB"), "cycle_q"] > 1.0
    assert keyed.loc[("B", "AB"), "cycle_weight_factor"] < 1.0


def test_fit_node_potentials_uses_reverse_observation_as_separate_row():
    edges = pd.DataFrame(
        [
            {"from_mutation": "WT", "to_mutation": "A", "calc_ddG": 1.0, "calc_ddG_err": 0.1},
            {"from_mutation": "A", "to_mutation": "WT", "calc_ddG": -3.0, "calc_ddG_err": 0.1},
        ]
    )

    fit = fit_node_potentials(edges, method="wls")
    energy = dict(zip(fit.nodes["node"], fit.nodes["energy"]))

    # The two observations imply d=1 and d=3 for E_A-E_WT, so equal weights give d=2.
    assert energy["A"] == pytest.approx(2.0)


def test_expand_bidirectional_edges_replaces_forward_with_two_observations():
    edges = pd.DataFrame(
        [
            {
                "source": "phase2_charge",
                "system": "1ABC",
                "from_mutation": "WT",
                "to_mutation": "A",
                "calc_ddG": 1.0,
                "calc_ddG_err": 0.2,
            },
            {
                "source": "phase2_charge",
                "system": "1ABC",
                "from_mutation": "WT",
                "to_mutation": "B",
                "calc_ddG": 0.5,
                "calc_ddG_err": 0.2,
            },
        ]
    )
    bidir = pd.DataFrame(
        [
            {
                "source": "phase2_charge",
                "system": "1ABC",
                "from_mutation": "WT",
                "to_mutation": "A",
                "ddG_f": 1.0,
                "ddG_f_err": 0.2,
                "ddG_r": -2.0,
                "ddG_r_err": 0.3,
                "hysteresis": -1.0,
                "bidir_avg": 1.5,
                "bidir_err": 0.18,
            }
        ]
    )

    expanded = expand_bidirectional_edges(edges, bidir)
    assert len(expanded) == 3
    assert set(expanded["bidir_observation"]) == {"none", "forward", "reverse"}
    reverse = expanded[expanded["bidir_observation"].eq("reverse")].iloc[0]
    assert reverse["from_mutation"] == "A"
    assert reverse["to_mutation"] == "WT"
    assert reverse["calc_ddG"] == pytest.approx(-2.0)


def _three_paths_with_one_outlier() -> pd.DataFrame:
    """Three consistent WT->AB pathways (each summing to 3.0) plus one bad edge.

    The direct WT->AB observation of 9.0 disagrees with every pathway, so a
    robust fit should discount it rather than split the disagreement evenly.
    """
    rows = [
        ("WT", "A", 1.0), ("A", "AB", 2.0),
        ("WT", "B", 2.0), ("B", "AB", 1.0),
        ("WT", "C", 1.5), ("C", "AB", 1.5),
        ("WT", "AB", 9.0),
    ]
    return pd.DataFrame(
        [{"from_mutation": f, "to_mutation": t, "calc_ddG": v} for f, t, v in rows]
    )


def test_huber_downweights_an_outlier_edge_that_wls_absorbs():
    edges = _three_paths_with_one_outlier()

    wls = fit_node_potentials(edges, method="wls", default_sigma=1.0)
    huber = fit_node_potentials(edges, method="huber", default_sigma=1.0, huber_delta=1.5)

    wls_energy = dict(zip(wls.nodes["node"], wls.nodes["energy"]))
    huber_energy = dict(zip(huber.nodes["node"], huber.nodes["energy"]))

    # Least squares spreads the 9.0 observation over every node.
    assert wls_energy["AB"] == pytest.approx(5.4)
    # IRLS pulls the estimate back towards the value the three pathways agree on.
    assert huber_energy["AB"] == pytest.approx(4.0)
    assert abs(huber_energy["AB"] - 3.0) < abs(wls_energy["AB"] - 3.0)
    assert huber.converged and huber.n_iter > 1

    factors = huber.edges.set_index(["from_mutation", "to_mutation"])[
        "node_fit_robust_factor"
    ]
    assert factors.loc[("WT", "AB")] == pytest.approx(0.3)
    consistent = factors.drop(index=("WT", "AB"))
    assert consistent.tolist() == pytest.approx([1.0] * len(consistent))


def test_wls_applies_no_robust_downweighting():
    """The WLS branch must not run IRLS, so every edge keeps unit weight."""
    fit = fit_node_potentials(
        _three_paths_with_one_outlier(), method="wls", default_sigma=1.0
    )

    assert fit.converged and fit.n_iter == 1
    factors = fit.edges["node_fit_robust_factor"].tolist()
    assert factors == pytest.approx([1.0] * len(factors))


def test_fit_node_potentials_returns_empty_when_reference_is_absent():
    """A reference node outside the graph yields no component to fit."""
    edges = pd.DataFrame(
        [{"from_mutation": "X", "to_mutation": "Y", "calc_ddG": 1.0}]
    )

    fit = fit_node_potentials(edges, ref_node="WT", method="huber")

    assert fit.nodes.empty
    assert fit.edges.empty
    assert not fit.converged


def test_fit_node_potentials_rejects_an_unknown_method():
    with pytest.raises(ValueError, match="method must be"):
        fit_node_potentials(_three_paths_with_one_outlier(), method="nonesuch")
