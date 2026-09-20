#!/usr/bin/env python3
"""Collect FEP results and evaluate additive, WCC and stepwise estimates."""

import sys
import json
import argparse
import numpy as np
import pandas as pd
import networkx as nx
import pickle
from pathlib import Path

MUTMAP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MUTMAP_ROOT))

from protmutmap.tools import MutationList
from protmutmap.tools.lambda_calculator import assess_mutation_difficulty
from protmutmap.gather_results import gather_results
from protmutmap.wcc.main import wcc_from_dataframe
from protmutmap.bidirectional import (
    apply_bidirectional_to_calc_df,
    build_bidir_lookup,
    compute_bidirectional_table,
)

# Systems dict: system_name → {target_id, entries}

SYSTEMS = {

    "1A22": {
        "target_id": "1A22_A_B",
        "entries": [
            "FA25A,YA42A,QA46A",
        ],
    },
    "1AO7": {
        "target_id": "1AO7_ABC_DE",
        "entries": [
            "AE96M,GE97S,GE98A",
            "GD28M,SD51M,SD94T",
            "GD28T,AE96M,GE97S,GE98A",
            "GD28T,SD51M,SD94T",
        ],
    },
    "1BJ1": {
        "target_id": "1BJ1_HL_VW",
        "entries": [
            "HH101Y,YH103W,SH105T",
        ],
    },
    "2B2X": {
        "target_id": "2B2X_HL_A",
        "entries": [
            "VH50T,QL28S,YL52N",
        ],
    },
    "2J0T": {
        "target_id": "2J0T_A_D",
        "entries": [
            "TD2L,VD4S,SD68A",
            "TD2S,VD4A,SD68Y",
        ],
    },

    "1CHO": {
        "target_id": "1CHO_EFG_I",
        "entries": [
            "TI14M,LI15S,GI29V",
        ],
    },
    "1R0R": {
        "target_id": "1R0R_E_I",
        "entries": [
            "TI12M,LI13S,GI27V",
        ],
    },
    "3SGB": {
        "target_id": "3SGB_E_I",
        "entries": [
            "TI11M,LI12S,GI26V",
        ],
    },
    "4OFY": {
        "target_id": "4OFY_A_D",
        "entries": [
            "QA34A,MA36A,FA40A,SA87A",
        ],
    },

    "1AHW": {
        "target_id": "1AHW_AB_C",
        "entries": [
            "IC141Q,TC143I,YC145T",
        ],
    },
    "3SE3": {
        "target_id": "3SE3_B_A",
        "entries": [
            "YB43H,NB44A,SB47Q",
            "YB43A,NB44A,SB47A",
        ],
    },
    "3EG5": {
        "target_id": "3EG5_A_B",
        "entries": [
            "TB73N,SB74N,HB75N",
        ],
    },
    "1MHP": {
        "target_id": "1MHP_HL_A",
        "entries": [
            "TH33I,SH52T,GH53N,GH54N",
        ],
    },
}

SYSTEM_CATEGORY = {

    "1A22": "Pathway",
    "1AO7": "Pathway",
    "1BJ1": "Pathway",
    "2B2X": "Pathway",
    "2J0T": "Pathway",
    "1CHO": "SingleRef",
    "1R0R": "SingleRef",
    "3SGB": "SingleRef",
    "4OFY": "SingleRef",
    "1AHW": "TargetOnly",
    "3SE3": "TargetOnly",
    "3EG5": "TargetOnly",
    "1MHP": "TargetOnly",
}

BASE_METHODS = ["Additive", "WCC", "Stepwise"]

EXTRA_METHODS = [
    # charge-edge ablation
    "WCC_no_charge",
    "Stepwise_no_charge",
    # edge-uncertainty inflated WCC
    "WCC_charge_weighted",
    "WCC_cycle_weighted",
    # Sampson/Rocklin analytical charge correction (PBC + undersolvation)
    "WCC_charge_corrected",
    "Stepwise_charge_corrected",
    # bidirectional-aware variants. Charge edges use (ddG_f − ddG_r)/2;
    # all non-charge edges fall back to the forward FEP value, so on systems with
    # no reverse FEPs these collapse to Additive / WCC / Stepwise.
    "Additive_bidirectional",
    "WCC_bidirectional",
    "Stepwise_bidirectional",
]

EXTENDED_METHODS = BASE_METHODS + EXTRA_METHODS
ALL_METHODS = list(EXTENDED_METHODS)  # may be further extended by --extra-networks at runtime

METHOD_LABELS = {
    "Additive": "Additive",
    "WCC": "WCC",
    "Stepwise": "Stepwise",
    "WCC_no_charge": "WCC (without charge edges)",
    "Stepwise_no_charge": "Stepwise (without charge edges)",
    "WCC_charge_weighted": "WCC (charge weighting)",
    "WCC_cycle_weighted": "WCC (cycle weighting)",
    "WCC_charge_corrected": "WCC (charge correction)",
    "Stepwise_charge_corrected": "Stepwise (charge correction)",
    "Additive_bidirectional": "Additive (bidirectional)",
    "WCC_bidirectional": "WCC (bidirectional)",
    "Stepwise_bidirectional": "Stepwise (bidirectional)",
}

def get_partner_chains(target_id: str) -> dict:
    parts = target_id.split("_")
    return {
        "partner1": list(parts[1]),
        "partner2": list(parts[2]),
    }

def get_single_mut_ddg(
    single_mut_str: str,
    base_target_dir: Path,
    partner_chains_dict: dict,
    *,
    time_ps: float | None = None,
) -> tuple[float, float]:
    """
    Read WT→single_mutation FEP result (ddG = complex_dG - partner1_dG - partner2_dG).
    Returns (val, err).
    """
    row = pd.Series({"from_mutation": "WT", "to_mutation": single_mut_str, "mutation_diff": single_mut_str})
    df_one = pd.DataFrame([row])
    result_df = gather_results(df_one, base_target_dir, partner_chains_dict, time_ps=time_ps)
    if result_df.empty or "calc_ddG" not in result_df.columns:
        return np.nan, np.nan
    val = float(result_df["calc_ddG"].iloc[0])
    err = float(result_df["calc_ddG_err"].iloc[0]) if "calc_ddG_err" in result_df.columns else np.nan
    return val, err

def estimate_additive(
    target_entry: str,
    base_target_dir: Path,
    partner_chains_dict: dict,
    *,
    time_ps: float | None = None,
) -> tuple[float, float]:
    """
    Pattern a: sum of single-mutation ddGs from WT structure.
    Returns (val, err) where err = sqrt(sum of individual errs squared).
    """
    muts = MutationList.from_string(target_entry)
    total = 0.0
    total_err_sq = 0.0
    for single_str in muts.to_list():
        ddg, err = get_single_mut_ddg(single_str, base_target_dir, partner_chains_dict, time_ps=time_ps)
        if np.isnan(ddg):
            return np.nan, np.nan
        total += ddg
        if not np.isnan(err):
            total_err_sq += err ** 2
    return total, np.sqrt(total_err_sq) if total_err_sq > 0 else 0.0

def estimate_additive_bidirectional(
    target_entry: str,
    base_target_dir: Path,
    partner_chains_dict: dict,
    bidir_lookup: dict,
    *,
    time_ps: float | None = None,
) -> tuple[float, float]:
    """Pattern a with bidirectional-aware lookup for charge-changing single mutations.

    For each single mutation S in `target_entry`, if (WT, S) appears in
    `bidir_lookup` we use `bidir_avg` / `bidir_err`; otherwise we fall back to
    the forward FEP via `get_single_mut_ddg`.
    """
    if not bidir_lookup:
        return estimate_additive(target_entry, base_target_dir, partner_chains_dict, time_ps=time_ps)
    muts = MutationList.from_string(target_entry)
    total = 0.0
    total_err_sq = 0.0
    for single_str in muts.to_list():
        key = ("WT", single_str)
        if key in bidir_lookup:
            ddg, err = bidir_lookup[key]
        else:
            ddg, err = get_single_mut_ddg(single_str, base_target_dir, partner_chains_dict, time_ps=time_ps)
        if np.isnan(ddg):
            return np.nan, np.nan
        total += ddg
        if not np.isnan(err):
            total_err_sq += err ** 2
    return total, np.sqrt(total_err_sq) if total_err_sq > 0 else 0.0

def run_wcc(calc_result_df: pd.DataFrame) -> tuple[dict, dict]:
    """Run WCC (unweighted) and return ({mutation_str: predicted_ddG}, {mutation_str: path_dep_error})."""
    valid_df = calc_result_df.dropna(subset=["calc_ddG"])
    if valid_df.empty:
        return {}, {}
    valid_df = _keep_wt_connected(valid_df)
    if valid_df.empty:
        return {}, {}
    result = wcc_from_dataframe(
        valid_df[["from_mutation", "to_mutation", "calc_ddG"]],
        ref_mol="WT",
        mol1_col="from_mutation",
        mol2_col="to_mutation",
        ddg_col="calc_ddG",
    )
    if result is None:
        return {}, {}
    energies = dict(zip(result["molecules"], result["energies"][0]))
    path_dep_errors = dict(zip(result["molecules"], result["path_dependent_errors"]))
    return energies, path_dep_errors

def run_wcc_weighted(
    calc_result_df: pd.DataFrame,
    suspect_score: pd.Series | None = None,
    *,
    alpha: float = 2.0,
    boolean_flag_col: str | None = None,
    ddg_col: str = "calc_ddG",
    err_col: str = "calc_ddG_err",
) -> tuple[dict, dict]:
    """Run WCC with inflated per-edge uncertainty.

    The Graphs.py CycleClosure already distributes cycle delta by `weight/std_sum`,
    where weight = err**2 when uncertainty_col is provided. Inflating err on
    suspect edges therefore makes those edges absorb proportionally more of the
    cycle correction, leaving consensus edges less perturbed.

    Two modes:
      - boolean_flag_col: inflate err by `1 + alpha` for rows where the flag is True
        (e.g. has_charge_change). Used for WCC_charge_weighted.
      - suspect_score: pd.Series indexed by (from, to) tuple with a non-negative
        score; inflate err by `1 + alpha * score`. Used for WCC_cycle_weighted with
        edge_cycle_score.

    Weights modify cycle closure; they do not perform a Huber fit."""
    valid_df = calc_result_df.dropna(subset=[ddg_col]).copy()
    if valid_df.empty or err_col not in valid_df.columns:
        return {}, {}

    # Inflated uncertainty column
    base_err = valid_df[err_col].astype(float).fillna(0.0)
    if boolean_flag_col is not None and boolean_flag_col in valid_df.columns:
        flag = valid_df[boolean_flag_col].fillna(False).astype(bool)
        inflate = pd.Series(np.where(flag, 1.0 + alpha, 1.0), index=valid_df.index)
        valid_df["_err_inflated"] = base_err * inflate
    elif suspect_score is not None:
        # Lookup score per (from, to); missing keys -> 0
        scores = []
        for _, row in valid_df.iterrows():
            key = (row["from_mutation"], row["to_mutation"])
            scores.append(float(suspect_score.get(key, 0.0)))
        scores_arr = np.asarray(scores, dtype=float)
        valid_df["_err_inflated"] = base_err * (1.0 + alpha * scores_arr)
    else:
        valid_df["_err_inflated"] = base_err

    valid_df = _keep_wt_connected(valid_df)
    if valid_df.empty:
        return {}, {}

    result = wcc_from_dataframe(
        valid_df[["from_mutation", "to_mutation", ddg_col, "_err_inflated"]],
        ref_mol="WT",
        mol1_col="from_mutation",
        mol2_col="to_mutation",
        ddg_col=ddg_col,
        uncertainty_col="_err_inflated",
    )
    if result is None:
        return {}, {}
    # result["energies"] is a list of arrays; index 0 = unweighted, index 1 = weighted.
    # When uncertainty_col is provided we want the weighted energies (index 1).
    energies_list = result["energies"]
    if len(energies_list) >= 2:
        energies = dict(zip(result["molecules"], energies_list[1]))
    else:
        energies = dict(zip(result["molecules"], energies_list[0]))
    path_dep_errors = dict(zip(result["molecules"], result["path_dependent_errors"]))
    return energies, path_dep_errors

def _keep_wt_connected(df: pd.DataFrame) -> pd.DataFrame:
    """
    Restrict DataFrame to edges whose endpoints are all reachable from WT.

    After filtering out high-error edges, some nodes may become disconnected
    from WT. This helper ensures the WCC graph remains connected to WT, so
    that Dijkstra path reconstruction in CalLig.py can safely traverse all edges.
    """
    G = nx.Graph()
    for _, row in df.iterrows():
        G.add_edge(row["from_mutation"], row["to_mutation"])
    if not G.has_node("WT"):
        return df  # can't determine reachability; return as-is
    reachable = nx.node_connected_component(G, "WT")
    return df[df["from_mutation"].isin(reachable) & df["to_mutation"].isin(reachable)]

def _is_step_difficult(from_node: str, to_node: str) -> bool:
    """
    Check if the mutation step from from_node to to_node involves a difficult
    residue (F/Y/W/P, with FY<->FY exception). Uses mutation labels encoded
    in node names (e.g., "FA25A" means F->A at chain A, residue 25).
    """
    from_set = set() if from_node == "WT" else set(from_node.split(","))
    to_set = set() if to_node == "WT" else set(to_node.split(","))
    changed_muts = (from_set - to_set) | (to_set - from_set)
    for mut_str in changed_muts:
        try:
            muts = MutationList.from_string(mut_str)
            is_diff, _, _ = assess_mutation_difficulty(muts)
            if is_diff:
                return True
        except Exception:
            pass
    return False

def estimate_stepwise(
    target_entry: str,
    graph: nx.DiGraph,
    calc_result_df: pd.DataFrame,
) -> tuple[float, float]:
    """
    Pattern d/e: shortest path from WT to target_entry.
    When multiple shortest paths exist, prefer the one where difficult
    mutations (F/Y/W/P) appear earliest (closest to WT structure).
    Returns (val, err) where err = sqrt(sum of edge errs squared) along best_path.
    """
    if not graph.has_node(target_entry):
        return np.nan, np.nan

    try:
        paths = list(nx.all_shortest_paths(graph, source="WT", target=target_entry))
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return np.nan, np.nan

    def difficulty_score(path: list) -> int:
        """Lower score = difficult edges appear earlier in path (preferred)."""
        return sum(
            i * _is_step_difficult(path[i], path[i + 1])
            for i in range(len(path) - 1)
        )

    best_path = min(paths, key=difficulty_score)

    # Build lookup for edge ddG and err: (from, to) -> calc_ddG / calc_ddG_err
    edge_ddg = {}
    edge_err = {}
    for _, row in calc_result_df.iterrows():
        fwd = (row["from_mutation"], row["to_mutation"])
        rev = (row["to_mutation"], row["from_mutation"])
        ddg_val = row.get("calc_ddG", np.nan)
        err_val = row.get("calc_ddG_err", np.nan)
        edge_ddg[fwd] = ddg_val
        edge_err[fwd] = err_val
        if rev not in edge_ddg:
            edge_ddg[rev] = -ddg_val if not np.isnan(ddg_val) else np.nan
            edge_err[rev] = err_val  # error is symmetric

    total = 0.0
    total_err_sq = 0.0
    for i in range(len(best_path) - 1):
        key = (best_path[i], best_path[i + 1])
        if key not in edge_ddg or np.isnan(edge_ddg[key]):
            return np.nan, np.nan
        total += edge_ddg[key]
        e = edge_err.get(key, np.nan)
        if not np.isnan(e):
            total_err_sq += e ** 2
    return total, np.sqrt(total_err_sq) if total_err_sq > 0 else 0.0

def _compute_all_wcc_variants(
    calc_df: pd.DataFrame,
    edge_cycle_score: pd.Series | None = None,
) -> dict:
    """Compute WCC and weighted variants for a calc_df.

    Returns dict with keys:
        wcc, wcc_err               -- standard unweighted WCC (WCC family)
        wcc_charge_w, ..._err      -- inflate has_charge_change=True edge errs by alpha=2
        wcc_cycle_w, ..._err       -- inflate err by edge_cycle_score"""
    out = {
        "wcc": {}, "wcc_err": {},
        "wcc_charge_w": {}, "wcc_charge_w_err": {},
        "wcc_cycle_w": {}, "wcc_cycle_w_err": {},
    }
    if calc_df is None or calc_df.empty:
        return out
    out["wcc"], out["wcc_err"] = run_wcc(calc_df)
    if "has_charge_change" in calc_df.columns:
        out["wcc_charge_w"], out["wcc_charge_w_err"] = run_wcc_weighted(
            calc_df, boolean_flag_col="has_charge_change", alpha=2.0,
        )
    if edge_cycle_score is not None:
        out["wcc_cycle_w"], out["wcc_cycle_w_err"] = run_wcc_weighted(
            calc_df, suspect_score=edge_cycle_score, alpha=2.0,
        )
    return out

def _filter_charge_edges(calc_df: pd.DataFrame) -> pd.DataFrame:
    """Drop edges marked has_charge_change=True. Used to ablate charge-change edges
    from WCC/SP for diagnostic patterns WCC_no_charge / Stepwise_no_charge.
    Targets unreachable after the drop will get NaN from the downstream estimators.
    """
    if calc_df is None or calc_df.empty or "has_charge_change" not in calc_df.columns:
        return calc_df
    return calc_df[~calc_df["has_charge_change"].fillna(False).astype(bool)].copy()

def _build_sampson_corrected_calc_df(
    calc_df: pd.DataFrame,
    base_target_dir: Path,
    partner_chains_dict: dict,
    *,
    epsS: float = 78.0,
) -> pd.DataFrame | None:
    """Apply Sampson/Rocklin analytical correction  to charge-change edges.

    Reads the actual box dimensions from each FEP working directory's
    conf_ionized.pdb and adjusts calc_ddG for has_charge_change=True edges.
    Edges without charge change are passed through unchanged.

    Returns a copy of calc_df with `calc_ddG` replaced by the corrected value
    (and `calc_ddG_orig`, `correction_ddG_kcal`, `delta_q` added). Returns None
    if no charge edges are present."""
    if calc_df is None or calc_df.empty:
        return None
    if "has_charge_change" not in calc_df.columns:
        return None
    if not calc_df["has_charge_change"].fillna(False).astype(bool).any():
        return None

    from protmutmap.charge_correction import (
        apply_corrections_to_edges_df,
    )
    from protmutmap.gather_results import _process_single_edge

    # Build per-edge per-partner box length lookup by walking the FEP work dirs.
    box_lookup: dict[tuple[str, str, str], float] = {}
    for _, row in calc_df.iterrows():
        if not bool(row.get("has_charge_change", False)):
            continue
        edge_files = _process_single_edge(row, base_target_dir, partner_chains_dict)
        for mode, file_path in edge_files.items():
            if file_path is None:
                continue
            for candidate in ("conf_ionized.pdb", "conf_solvated.pdb", "conf_box.pdb"):
                pdb_path = Path(file_path) / candidate
                if pdb_path.exists():
                    from protmutmap.charge_correction import read_box_lref_from_pdb
                    L = read_box_lref_from_pdb(pdb_path)
                    if L is not None:
                        box_lookup[(row["from_mutation"], row["to_mutation"], mode)] = L
                        break

    corrected = apply_corrections_to_edges_df(
        calc_df,
        box_length_lookup=box_lookup,
        default_box_nm=8.0,
        epsS=epsS,
        only_charge_edges=True,
    )
    # Replace calc_ddG with corrected for downstream WCC/SP consumers
    corrected = corrected.rename(columns={"calc_ddG": "calc_ddG_orig"})
    corrected["calc_ddG"] = corrected["calc_ddG_corrected"]
    return corrected

def _build_no_charge_graph(graph: nx.DiGraph | None, calc_df: pd.DataFrame) -> nx.DiGraph | None:
    """Return a copy of `graph` with edges that are charge-change removed.

    Used for Stepwise_no_charge so SP avoids the same edges WCC ignores. Mirrors
    the row-level has_charge_change filter applied to calc_df.
    """
    if graph is None or calc_df is None or calc_df.empty:
        return graph
    if "has_charge_change" not in calc_df.columns:
        return graph
    G = graph.copy()
    for _, row in calc_df.iterrows():
        if bool(row.get("has_charge_change", False)):
            u, v = row["from_mutation"], row["to_mutation"]
            for edge in ((u, v), (v, u)):
                if G.has_edge(*edge):
                    G.remove_edge(*edge)
    return G

def _merge_calc_dfs(df_primary: pd.DataFrame, df_secondary: pd.DataFrame) -> pd.DataFrame:
    """
    Merge two calc_df DataFrames by (from_mutation, to_mutation).
    df_primary takes precedence; df_secondary fills in missing edges.
    """
    if df_primary is None or df_primary.empty:
        return df_secondary if (df_secondary is not None) else pd.DataFrame()
    if df_secondary is None or df_secondary.empty:
        return df_primary

    idx_cols = ["from_mutation", "to_mutation"]
    merged = (
        df_primary.set_index(idx_cols)
        .combine_first(df_secondary.set_index(idx_cols))
        .reset_index()
    )
    return merged

def _build_row(
    node: str,
    exp_ddg: float,
    system_name: str,
    target_id: str,
    node_type: str,
    additive_value: float,
    additive_value_err: float,
    additive_value_bidir: float,
    additive_value_bidir_err: float,
    wt: dict,
    calc_df_wt: pd.DataFrame,
    graph: nx.DiGraph | None,
    partner_chains_dict: dict,
    extra_results: dict | None = None,
    system_category_map: dict | None = None,
) -> dict:
    """Build a result row for one node (target or intermediate).

    extra_results: dict of name → {"wcc": ..., "wcc_err": ..., "graph": ..., "calc_df": ...}
    """
    # WCC-WT pattern
    wcc_value = wt["wcc"].get(node, np.nan)
    wcc_value_err = wt["wcc_err"].get(node, np.nan)
    # Shortest-path patterns
    stepwise_value, stepwise_value_err = estimate_stepwise(node, graph, calc_df_wt) if graph else (np.nan, np.nan)

    # Charge-edge ablation: WCC/SP on the graph with all
    # has_charge_change=True edges dropped. Targets unreachable from WT after the
    # ablation get NaN. Computed unconditionally — for systems with no charge edges
    # these collapse to the regular WCC / Stepwise.
    wcc_value_no_charge, wcc_value_no_charge_err = np.nan, np.nan
    stepwise_value_no_charge, stepwise_value_no_charge_err = np.nan, np.nan
    if calc_df_wt is not None and not calc_df_wt.empty and "has_charge_change" in calc_df_wt.columns:
        ablated_calc = _filter_charge_edges(calc_df_wt)
        if not ablated_calc.empty:
            ablated_wcc = _compute_all_wcc_variants(ablated_calc)
            wcc_value_no_charge = ablated_wcc["wcc"].get(node, np.nan)
            wcc_value_no_charge_err = ablated_wcc["wcc_err"].get(node, np.nan)
            ablated_graph = _build_no_charge_graph(graph, calc_df_wt)
            stepwise_value_no_charge, stepwise_value_no_charge_err = (
                estimate_stepwise(node, ablated_graph, ablated_calc)
                if ablated_graph else (np.nan, np.nan)
            )

    # WCC with inflated uncertainty on suspect edges (charge / cycle).
    wcc_value_charge_w = wt.get("wcc_charge_w", {}).get(node, np.nan)
    wcc_value_charge_w_err = wt.get("wcc_charge_w_err", {}).get(node, np.nan)
    wcc_value_cycle_w = wt.get("wcc_cycle_w", {}).get(node, np.nan)
    wcc_value_cycle_w_err = wt.get("wcc_cycle_w_err", {}).get(node, np.nan)

    # Sampson/Rocklin charge correction (analytical PBC + undersolvation)
    # Computed once per system, cached in `wt` dict by gather_system.
    wcc_value_sampson = wt.get("wcc_sampson", {}).get(node, np.nan)
    wcc_value_sampson_err = wt.get("wcc_sampson_err", {}).get(node, np.nan)
    stepwise_value_sampson = np.nan
    stepwise_value_sampson_err = np.nan
    sampson_calc_df = wt.get("_sampson_calc_df")
    if (
        sampson_calc_df is not None
        and not sampson_calc_df.empty
        and graph is not None
    ):
        stepwise_value_sampson, stepwise_value_sampson_err = estimate_stepwise(node, graph, sampson_calc_df)

    # Combine paired directions as (ddG_f − ddG_r) / 2, with
    # error sqrt(σ_f² + σ_r²) / 2. Unpaired edges retain their forward values.
    wcc_value_bidir = wt.get("wcc_bidir", {}).get(node, np.nan)
    wcc_value_bidir_err = wt.get("wcc_bidir_err", {}).get(node, np.nan)
    if (isinstance(wcc_value_bidir, float) and np.isnan(wcc_value_bidir)) or wcc_value_bidir is None:
        wcc_value_bidir = wcc_value
        wcc_value_bidir_err = wcc_value_err
    stepwise_value_bidir = np.nan
    stepwise_value_bidir_err = np.nan
    bidir_calc_df = wt.get("_bidir_calc_df")
    if (
        bidir_calc_df is not None
        and not bidir_calc_df.empty
        and graph is not None
    ):
        stepwise_value_bidir, stepwise_value_bidir_err = estimate_stepwise(node, graph, bidir_calc_df)
    if (isinstance(stepwise_value_bidir, float) and np.isnan(stepwise_value_bidir)) or stepwise_value_bidir is None:
        stepwise_value_bidir = stepwise_value
        stepwise_value_bidir_err = stepwise_value_err

    row = {
        "system": system_name,
        "target_id": target_id,
        "mutation": node,
        "mutation_num": len(node.split(",")),
        "category": (system_category_map or SYSTEM_CATEGORY).get(system_name, "Unknown"),
        "node_type": node_type,
        "wcc_leaf": False,  # overridden for isolated single-mut nodes in gather_intermediate_nodes
        "exp_ddG": exp_ddg,
        "Additive": additive_value,
        "WCC": wcc_value,
        "Stepwise": stepwise_value,
        "WCC_no_charge": wcc_value_no_charge,
        "Stepwise_no_charge": stepwise_value_no_charge,
        "WCC_charge_weighted": wcc_value_charge_w,
        "WCC_cycle_weighted": wcc_value_cycle_w,
        "WCC_charge_corrected": wcc_value_sampson,
        "Stepwise_charge_corrected": stepwise_value_sampson,
        "Additive_bidirectional": additive_value_bidir,
        "WCC_bidirectional": wcc_value_bidir,
        "Stepwise_bidirectional": stepwise_value_bidir,
        "Additive_err": additive_value_err,
        "WCC_err": wcc_value_err,
        "Stepwise_err": stepwise_value_err,
        "WCC_no_charge_err": wcc_value_no_charge_err,
        "Stepwise_no_charge_err": stepwise_value_no_charge_err,
        "WCC_charge_weighted_err": wcc_value_charge_w_err,
        "WCC_cycle_weighted_err": wcc_value_cycle_w_err,
        "WCC_charge_corrected_err": wcc_value_sampson_err,
        "Stepwise_charge_corrected_err": stepwise_value_sampson_err,
        "Additive_bidirectional_err": additive_value_bidir_err,
        "WCC_bidirectional_err": wcc_value_bidir_err,
        "Stepwise_bidirectional_err": stepwise_value_bidir_err,
    }

    # Extra network patterns
    if extra_results:
        for name, er in extra_results.items():
            wcc_dict = er.get("wcc", {})
            wcc_err_dict = er.get("wcc_err", {})
            eg = er.get("graph")
            edf = er.get("calc_df")

            b_val = wcc_dict.get(node, np.nan)
            b_err = wcc_err_dict.get(node, np.nan)
            d_val, d_err = (
                estimate_stepwise(node, eg, edf)
                if (eg is not None and edf is not None and not edf.empty)
                else (np.nan, np.nan)
            )
            row[f"WCC_{name}"] = b_val
            row[f"Stepwise_{name}"] = d_val
            row[f"WCC_{name}_err"] = b_err
            row[f"Stepwise_{name}_err"] = d_err

    return row

def gather_intermediate_nodes(
    exp_dgs: dict,
    target_entries: set,
    calc_df_wt: pd.DataFrame,
    wt: dict,
    graph: nx.DiGraph | None,
    system_name: str,
    target_id: str,
    base_target_dir: Path,
    partner_chains_dict: dict,
    extra_results: dict | None = None,
    system_category_map: dict | None = None,
    *,
    time_ps: float | None = None,
) -> list[dict]:
    """
    For each intermediate node in dGs.json (i.e., single or sub-combination
    mutations that are NOT themselves target entries), compute all patterns
    and return rows for comparison.

    """
    rows = []
    for mut_str, exp_ddg in exp_dgs.items():
        if mut_str == "WT":
            continue
        if mut_str in target_entries:
            continue

        n_muts = len(mut_str.split(","))
        node_type = "single" if n_muts == 1 else f"subcomb_{n_muts}mut"

        # Flag isolated single-mutation nodes: those with no downstream edges
        # (other than WT→X) that have FEP data. These are leaves in the WCC graph
        # and their WCC/SP value simply equals the single-edge FEP result.
        wcc_leaf = False
        if n_muts == 1 and calc_df_wt is not None and not calc_df_wt.empty:
            downstream = calc_df_wt[
                (calc_df_wt["from_mutation"] == mut_str)
                & calc_df_wt["calc_ddG"].notna()
            ]
            wcc_leaf = downstream.empty

        try:
            additive_value, additive_value_err = estimate_additive(mut_str, base_target_dir, partner_chains_dict, time_ps=time_ps)
        except Exception:
            additive_value, additive_value_err = np.nan, np.nan

        bidir_lookup = wt.get("_bidir_lookup", {}) if wt else {}
        try:
            additive_value_bidir_v, additive_value_bidir_err_v = estimate_additive_bidirectional(
                mut_str, base_target_dir, partner_chains_dict, bidir_lookup,
                time_ps=time_ps,
            )
        except Exception:
            additive_value_bidir_v, additive_value_bidir_err_v = np.nan, np.nan

        row = _build_row(
            node=mut_str,
            exp_ddg=exp_ddg,
            system_name=system_name,
            target_id=target_id,
            node_type=node_type,
            additive_value=additive_value,
            additive_value_err=additive_value_err,
            additive_value_bidir=additive_value_bidir_v,
            additive_value_bidir_err=additive_value_bidir_err_v,
            wt=wt,
            calc_df_wt=calc_df_wt,
            graph=graph,
            partner_chains_dict=partner_chains_dict,
            extra_results=extra_results,
            system_category_map=system_category_map,
        )
        row["wcc_leaf"] = wcc_leaf
        rows.append(row)

    return rows

def gather_system(
    system_name: str,
    system_info: dict,
    base_dir: Path,
    work_dir_name: str = "mutmap_work",
    extra_networks: list | None = None,
    network_dir_name: str = "network",
    dgs_filename: str = "dGs.json",
    system_category_map: dict | None = None,
    return_diagnostics: bool = False,
    *,
    time_ps: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame] | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Process one system and return (target_df, intermediate_df).

    target_df: results for primary target entries (multi-mutation)
    intermediate_df: results for intermediate nodes (single + sub-combination)

    extra_networks: list of (name, network_dir_name, work_dir_name) tuples.
      For each extra network:
        1. gather FEP from wt_work_dir using extra network's links (gets standard edges)
        2. gather FEP from extra work_dir using extra network's links (gets new multi-mut edges)
        3. merge (extra work_dir takes priority) → compute WCC + SP
        → adds WCC_{name}, Stepwise_{name} columns
    """
    target_id = system_info["target_id"]
    partner_chains_dict = get_partner_chains(target_id)

    sys_dir = base_dir / "systems" / system_name
    network_dir = sys_dir / network_dir_name
    wt_work_dir = sys_dir / work_dir_name

    # Load experimental ddGs
    dgs_json = sys_dir / dgs_filename
    with open(dgs_json) as f:
        exp_dgs = json.load(f)

    # Load WT network
    links_tsv = network_dir / "links.tsv"
    node_tsv = network_dir / "node.tsv"
    graph_pkl = network_dir / "graph.pkl"

    if not links_tsv.exists():
        print(f"  WARNING: links.tsv not found for {system_name}, skipping")
        return pd.DataFrame(), pd.DataFrame()

    links_df = pd.read_csv(links_tsv, sep="\t")
    nodes_df = pd.read_csv(node_tsv, sep="\t")

    # Load WT graph for shortest-path patterns
    graph = None
    if graph_pkl.exists():
        with open(graph_pkl, "rb") as f:
            graph = pickle.load(f)

    # Gather WT-structure FEP results
    calc_df_wt = gather_results(links_df, wt_work_dir, partner_chains_dict, time_ps=time_ps)

    # Save edge-level FEP results (including BAR error) for analysis
    if not calc_df_wt.empty:
        links_ddg_path = network_dir / "links_with_ddg.tsv"
        save_cols = ["from_mutation", "to_mutation", "calc_ddG"]
        if "calc_ddG_err" in calc_df_wt.columns:
            save_cols.append("calc_ddG_err")
        calc_df_wt[save_cols].to_csv(links_ddg_path, sep="\t", index=False)

    edge_cycle_score_series: pd.Series | None = None
    edge_diagnostics_df: pd.DataFrame = pd.DataFrame()
    if not calc_df_wt.empty:
        try:
            from protmutmap.cycle_diagnostics import compute_edge_cycle_scores
            edge_diagnostics_df = compute_edge_cycle_scores(
                calc_df_wt,
                extra_cols=("has_charge_change", "is_difficult", "is_extreme"),
            )
            if not edge_diagnostics_df.empty:
                edge_cycle_score_series = pd.Series(
                    edge_diagnostics_df["edge_cycle_score"].values,
                    index=list(zip(
                        edge_diagnostics_df["from_mutation"],
                        edge_diagnostics_df["to_mutation"],
                    )),
                )
        except Exception as e:
            print(f"  WARNING: cycle diagnostics for {system_name} failed: {e}")

    wt = _compute_all_wcc_variants(
        calc_df_wt if not calc_df_wt.empty else None,
        edge_cycle_score=edge_cycle_score_series,
    )

    # bidirectional aggregation. Pair forward charge edges in
    # links_df with reverse FEPs run on the post-mutation state, compute
    # (ddG_f − ddG_r)/2 per charge edge, swap into a corrected calc_df, and
    # re-run WCC. SP for Stepwise_bidirectional is computed lazily in _build_row from
    # wt["_bidir_calc_df"]. wt["_bidir_lookup"] feeds Additive_bidirectional.
    bidir_table_df = pd.DataFrame()
    if not calc_df_wt.empty:
        try:
            bidir_table_df = compute_bidirectional_table(
                links_df, wt_work_dir, partner_chains_dict,
                time_ps=time_ps,
            )
        except Exception as e:
            print(f"  WARNING: bidirectional aggregation for {system_name}: {e}")
            bidir_table_df = pd.DataFrame()

    if not bidir_table_df.empty:
        bidir_calc_df = apply_bidirectional_to_calc_df(calc_df_wt, bidir_table_df)
        wt["_bidir_calc_df"] = bidir_calc_df
        wt["_bidir_lookup"] = build_bidir_lookup(bidir_table_df)
        wcc_bidir, wcc_bidir_err = run_wcc(bidir_calc_df)
        wt["wcc_bidir"] = wcc_bidir
        wt["wcc_bidir_err"] = wcc_bidir_err
    else:
        wt["_bidir_calc_df"] = None
        wt["_bidir_lookup"] = {}
        wt.setdefault("wcc_bidir", {})
        wt.setdefault("wcc_bidir_err", {})

    # Sampson/Rocklin charge correction (analytical only). Compute the
    # corrected calc_df once here and stash WCC + corrected df under wt["wcc_sampson"]
    # / wt["_sampson_calc_df"] for _build_row consumers.
    sampson_calc_df = _build_sampson_corrected_calc_df(
        calc_df_wt if not calc_df_wt.empty else None,
        wt_work_dir,
        partner_chains_dict,
    )
    if sampson_calc_df is not None and not sampson_calc_df.empty:
        wcc_sampson, wcc_sampson_err = run_wcc(sampson_calc_df)
        wt["wcc_sampson"] = wcc_sampson
        wt["wcc_sampson_err"] = wcc_sampson_err
        wt["_sampson_calc_df"] = sampson_calc_df
    else:
        wt.setdefault("wcc_sampson", {})
        wt.setdefault("wcc_sampson_err", {})
        wt["_sampson_calc_df"] = None
    extra_results = {}
    for (ename, enet_dir_name, ework_dir_name) in (extra_networks or []):
        extra_net_dir = sys_dir / enet_dir_name
        extra_work_dir = sys_dir / ework_dir_name

        if not extra_net_dir.exists():
            print(f"  INFO: extra network '{enet_dir_name}' not found for {system_name}, skipping")
            extra_results[ename] = {"wcc": {}, "wcc_err": {}, "graph": None, "calc_df": None}
            continue

        # Load extra network links and graph
        extra_links_tsv = extra_net_dir / "links.tsv"
        extra_graph_pkl = extra_net_dir / "graph.pkl"
        if not extra_links_tsv.exists():
            print(f"  WARNING: {enet_dir_name}/links.tsv not found for {system_name}, skipping")
            extra_results[ename] = {"wcc": {}, "wcc_err": {}, "graph": None, "calc_df": None}
            continue

        extra_links_df = pd.read_csv(extra_links_tsv, sep="\t")
        extra_graph = None
        if extra_graph_pkl.exists():
            with open(extra_graph_pkl, "rb") as f:
                extra_graph = pickle.load(f)
        calc_df_extra_wt = gather_results(extra_links_df, wt_work_dir, partner_chains_dict, time_ps=time_ps)
        calc_df_extra = None
        if extra_work_dir.exists():
            try:
                calc_df_extra = gather_results(extra_links_df, extra_work_dir, partner_chains_dict, time_ps=time_ps)
            except Exception as e:
                print(f"  WARNING: could not gather {ework_dir_name} for {system_name}: {e}")
        else:
            print(f"  INFO: extra work dir '{ework_dir_name}' not found for {system_name}, using WT only")

        # Extra-network values take priority; standard calculations fill missing edges.
        calc_df_extra = _merge_calc_dfs(calc_df_extra, calc_df_extra_wt)
        extra_wcc = _compute_all_wcc_variants(
            calc_df_extra if (calc_df_extra is not None and not calc_df_extra.empty) else None
        )
        extra_results[ename] = {
            "wcc": extra_wcc["wcc"],
            "wcc_err": extra_wcc["wcc_err"],
            "graph": extra_graph,
            "calc_df": calc_df_extra,
        }

    target_entries = set(system_info["entries"])
    target_rows = []
    for entry in system_info["entries"]:
        exp_ddg = exp_dgs.get(entry, np.nan)

        try:
            additive_value, additive_value_err = estimate_additive(entry, wt_work_dir, partner_chains_dict, time_ps=time_ps)
        except Exception as e:
            print(f"  WARNING Additive {entry}: {e}")
            additive_value, additive_value_err = np.nan, np.nan

        bidir_lookup = wt.get("_bidir_lookup", {})
        try:
            additive_value_bidir_v, additive_value_bidir_err_v = estimate_additive_bidirectional(
                entry, wt_work_dir, partner_chains_dict, bidir_lookup,
                time_ps=time_ps,
            )
        except Exception as e:
            print(f"  WARNING Additive_bidirectional {entry}: {e}")
            additive_value_bidir_v, additive_value_bidir_err_v = np.nan, np.nan

        row = _build_row(
            node=entry,
            exp_ddg=exp_ddg,
            system_name=system_name,
            target_id=target_id,
            node_type="target",
            additive_value=additive_value,
            additive_value_err=additive_value_err,
            additive_value_bidir=additive_value_bidir_v,
            additive_value_bidir_err=additive_value_bidir_err_v,
            wt=wt,
            calc_df_wt=calc_df_wt,
            graph=graph,
            partner_chains_dict=partner_chains_dict,
            extra_results=extra_results,
            system_category_map=system_category_map,
        )
        target_rows.append(row)
    intermediate_rows = gather_intermediate_nodes(
        exp_dgs=exp_dgs,
        target_entries=target_entries,
        calc_df_wt=calc_df_wt,
        wt=wt,
        graph=graph,
        system_name=system_name,
        target_id=target_id,
        base_target_dir=wt_work_dir,
        partner_chains_dict=partner_chains_dict,
        extra_results=extra_results,
        system_category_map=system_category_map,
        time_ps=time_ps,
    )

    target_df = pd.DataFrame(target_rows)
    intermediate_df = pd.DataFrame(intermediate_rows)

    if return_diagnostics:
        return target_df, intermediate_df, edge_diagnostics_df, bidir_table_df

    return target_df, intermediate_df

def compute_metrics(df: pd.DataFrame, col: str) -> dict:
    """Compute RMSE, MAE, R² for a prediction column against exp_ddG."""
    sub = df[["exp_ddG", col]].dropna()
    if len(sub) < 2:
        return {"rmse": np.nan, "mae": np.nan, "r2": np.nan, "n": len(sub)}
    exp = sub["exp_ddG"].values
    pred = sub[col].values
    rmse = np.sqrt(np.mean((exp - pred) ** 2))
    mae = np.mean(np.abs(exp - pred))
    ss_res = np.sum((exp - pred) ** 2)
    ss_tot = np.sum((exp - np.mean(exp)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {"rmse": rmse, "mae": mae, "r2": r2, "n": len(sub)}

def print_metrics_table(df: pd.DataFrame, label: str, methods: list) -> list[dict]:
    print(f"\n{label}")
    print(f"{'Method':<30} {'RMSE':>8} {'MAE':>8} {'R²':>8} {'N':>4}")
    print("-" * 62)
    rows = []
    for method in methods:
        if method not in df.columns:
            continue
        m = compute_metrics(df, method)
        m["method"] = method
        lbl = METHOD_LABELS.get(method, method)
        r2_str = f"{m['r2']:>8.3f}" if not np.isnan(m['r2']) else "     nan"
        rmse_str = f"{m['rmse']:>8.3f}" if not np.isnan(m['rmse']) else "     nan"
        mae_str = f"{m['mae']:>8.3f}" if not np.isnan(m['mae']) else "     nan"
        print(f"  {lbl:<28} {rmse_str} {mae_str} {r2_str} {m['n']:>4}")
        rows.append(m)
    return rows

def parse_extra_networks(specs: list[str]) -> list[tuple[str, str, str]]:
    """Parse list of 'NAME:NETWORK_DIR:WORK_DIR' strings."""
    result = []
    for spec in specs:
        parts = spec.split(":")
        if len(parts) != 3:
            raise ValueError(f"--extra-networks: expected NAME:NETWORK_DIR:WORK_DIR, got '{spec}'")
        result.append((parts[0], parts[1], parts[2]))
    return result

def main():
    parser = argparse.ArgumentParser(description="Gather all FEP results and compute patterns")
    parser.add_argument(
        "--base-dir",
        default=str(Path(__file__).resolve().parents[1]),
        help="experiments/ directory",
    )
    parser.add_argument(
        "--work-dir",
        default="mutmap_work",
        help="Work directory name for WT-structure FEP results (default: mutmap_work). "
             "Use mutmap_work_large for high-NREP recalculations.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: base-dir/results/ or base-dir/results_large/ for mutmap_work_large)",
    )
    parser.add_argument(
        "--extra-networks",
        nargs="+",
        default=[],
        metavar="NAME:NETWORK_DIR:WORK_DIR",
        help=(
            "Additional network+workdir combinations to evaluate as new patterns. "
            "Format: NAME:NETWORK_DIR:WORK_DIR (e.g. maxmut2:network_maxmut2:mutmap_work_maxmut2). "
            "For each entry, FEP data is gathered from both wt_work_dir (standard edges) "
            "and WORK_DIR (new multi-mutation edges), then merged for WCC/SP computation. "
            "Adds columns WCC_{NAME} and Stepwise_{NAME}."
        ),
    )
    parser.add_argument(
        "--time-ps",
        type=float,
        default=None,
        help="If set, read per-leg BAR estimates from bar_time_series.csv at "
             "this cutoff (ps) instead of the full-simulation bar1.log. When "
             "no explicit --output-dir is given, the default output dir gets "
             "a _<N>ns suffix (e.g. results_large → results_large_4ns) so "
             "the full-time results are not overwritten.",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    work_dir_name = args.work_dir
    time_ps = args.time_ps
    cutoff_suffix = "" if time_ps is None else f"_{int(round(time_ps / 1000))}ns"
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif work_dir_name == "mutmap_work_large":
        output_dir = base_dir / f"results_large{cutoff_suffix}"
    else:
        output_dir = base_dir / f"results{cutoff_suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)

    extra_networks = parse_extra_networks(args.extra_networks)

    for (ename, _, _) in extra_networks:
        for estimator in ("WCC", "Stepwise"):
            method = f"{estimator}_{ename}"
            if method not in ALL_METHODS:
                ALL_METHODS.append(method)
            if method not in METHOD_LABELS:
                METHOD_LABELS[method] = f"{estimator} ({ename})"

    all_target_rows = []
    all_intermediate_rows = []
    all_diagnostics_rows = []
    all_bidir_rows = []

    for system_name, system_info in SYSTEMS.items():
        print(f"\n{'='*60}")
        print(f"Processing: {system_name}")
        target_df, intermediate_df, diag_df, bidir_df = gather_system(
            system_name, system_info, base_dir, work_dir_name,
            extra_networks=extra_networks,
            return_diagnostics=True,
            time_ps=time_ps,
        )
        if not target_df.empty:
            all_target_rows.append(target_df)
        if not intermediate_df.empty:
            all_intermediate_rows.append(intermediate_df)
        if diag_df is not None and not diag_df.empty:
            d = diag_df.copy()
            d.insert(0, "system", system_name)
            all_diagnostics_rows.append(d)
        if bidir_df is not None and not bidir_df.empty:
            b = bidir_df.copy()
            b.insert(0, "system", system_name)
            all_bidir_rows.append(b)

    if not all_target_rows:
        print("No results found.")
        return

    target_df = pd.concat(all_target_rows, ignore_index=True)
    intermediate_df = pd.concat(all_intermediate_rows, ignore_index=True) if all_intermediate_rows else pd.DataFrame()

    # Combine all data points (targets + intermediates)
    all_df = pd.concat([target_df, intermediate_df], ignore_index=True) if not intermediate_df.empty else target_df

    target_csv = output_dir / "target_results.csv"
    target_df.to_csv(target_csv, index=False)
    print(f"\nSaved target results to {target_csv}")
    print(target_df.to_string())

    if not intermediate_df.empty:
        inter_csv = output_dir / "intermediate_results.csv"
        intermediate_df.to_csv(inter_csv, index=False)
        print(f"\nSaved intermediate results to {inter_csv}")
        print(intermediate_df.to_string())

    all_csv = output_dir / "all_results.csv"
    all_df.to_csv(all_csv, index=False)
    print(f"\nSaved combined results to {all_csv}")

    # Edge-level cycle diagnostics : per-edge cycle hysteresis scores
    # NOTE: identifiability — edges in the same cycle group score identically. The
    # `cycle_score_tied_n` column flags this. Do not use edge_cycle_score alone to
    # localize a single bad edge.
    if all_diagnostics_rows:
        diag_df_all = pd.concat(all_diagnostics_rows, ignore_index=True)
        diag_csv = output_dir / "edge_diagnostics.csv"
        diag_df_all.to_csv(diag_csv, index=False)
        print(f"\nSaved edge cycle diagnostics to {diag_csv}")

    # bidirectional FEP table — paired forward/reverse charge edges
    # with hysteresis (= ddG_f + ddG_r) and best-estimate (= (ddG_f − ddG_r)/2).
    if all_bidir_rows:
        bidir_df_all = pd.concat(all_bidir_rows, ignore_index=True)
        bidir_csv = output_dir / "edge_bidirectional.csv"
        bidir_df_all.to_csv(bidir_csv, index=False)
        print(f"\nSaved bidirectional charge-edge table to {bidir_csv}")

    # Summary metrics
    print("\n" + "=" * 60)
    metrics_rows = []

    # 1. Target entries only
    metrics_rows += print_metrics_table(target_df, "Metrics — Target entries only:", ALL_METHODS)

    # 2. All data points (targets + intermediates)
    if not intermediate_df.empty:
        metrics_rows += print_metrics_table(all_df, "Metrics — All data points (targets + intermediates):", ALL_METHODS)

    # Save metrics
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_csv = output_dir / "metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"\nSaved metrics to {metrics_csv}")

if __name__ == "__main__":
    main()
