#!/usr/bin/env python
"""Run robust node-potential estimators on an existing ProtMutMap result directory.

The script reads target/all node tables plus edge diagnostics, adds new robust
graph estimates, and writes all outputs to a separate directory.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache") / "matplotlib"))

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from protmutmap.robust_graph import cycle_residual_edge_scores, fit_node_potentials
from protmutmap.wcc.main import wcc_from_dataframe

BASE_METHODS = [
    ("Additive", "Approximate"),
    ("Stepwise", "Stepwise"),
    ("WCC", "WCC"),
    ("WCC_bidirectional", "WCC-bidir"),
]

ROBUST_METHODS = [
    ("node_bidir_multi_wls_unit", "Bidir Multi WLS"),
    ("node_bidir_multi_huber_unit", "Bidir Multi Huber"),
    ("node_bidir_multi_cycle_huber_a010", "CWNE alpha=.10"),
    ("node_bidir_multi_cycle_huber_a025", "CWNE alpha=.25"),
    ("node_bidir_multi_cycle_huber_a100", "CWNE"),
]

WCC_CYCLE_METHODS = [
    ("wcc_bidir_cycle_sigma_eff_a100", "Cycle-WCC-bidir sigma"),
    ("wcc_bidir_cycle_pseudo_var_a100", "Cycle-WCC-bidir var"),
]

PLOT_METHODS = BASE_METHODS + WCC_CYCLE_METHODS + ROBUST_METHODS
CYCLE_SWEEP_ALPHAS = [0.0, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
COMPARE_A = "WCC_bidirectional"
COMPARE_B = "node_bidir_multi_cycle_huber_a100"
COMPARE_A_LABEL = "WCC-bidir"
COMPARE_B_LABEL = "CWNE"
BAR_DEPENDENT_METHODS = [
    ("node_wls", "Node WLS"),
    ("node_huber", "Node Huber"),
    ("node_cycle_huber", "Cycle+Huber"),
    ("node_bidir_wls", "Bidir Node WLS"),
    ("node_bidir_huber", "Bidir Node Huber"),
    ("node_bidir_cycle_huber", "Bidir Cycle+Huber"),
    ("node_bidir_multi_wls", "Bidir Multi WLS"),
    ("node_bidir_multi_huber", "Bidir Multi Huber"),
    ("node_bidir_multi_cycle_huber", "Bidir Multi Cycle+Huber"),
]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        default="results/source_data/fep",
        help="Existing ProtMutMap result directory.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/analysis_robust_graph",
        help="Separate output directory for robust graph estimates and figures.",
    )
    parser.add_argument("--huber-delta", type=float, default=1.5)
    parser.add_argument("--bar-free-scale", type=float, default=1.0)
    parser.add_argument("--max-alt-path-len", type=int, default=6)
    parser.add_argument("--max-paths-per-edge", type=int, default=64)
    parser.add_argument("--min-sigma", type=float, default=0.05)
    return parser.parse_args()

def read_inputs(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target_df = pd.read_csv(input_dir / "target_results.csv")
    all_df = pd.read_csv(input_dir / "all_results.csv")
    edge_df = pd.read_csv(input_dir / "edge_diagnostics.csv")
    bidir_path = input_dir / "edge_bidirectional.csv"
    bidir_df = pd.read_csv(bidir_path) if bidir_path.exists() else pd.DataFrame()
    return target_df, all_df, edge_df, bidir_df

def apply_bidirectional_edges(edge_df: pd.DataFrame, bidir_df: pd.DataFrame) -> pd.DataFrame:
    out = edge_df.copy()
    out["bidir_replaced"] = False
    out["bidir_mode"] = "none"
    out["bidir_observation"] = "none"
    if bidir_df.empty:
        return out

    key_cols = ["source", "system", "from_mutation", "to_mutation"]
    cols = key_cols + ["bidir_avg", "bidir_err", "hysteresis"]
    merged = out.merge(bidir_df[cols], on=key_cols, how="left")
    mask = merged["bidir_avg"].notna()
    merged.loc[mask, "calc_ddG"] = merged.loc[mask, "bidir_avg"]
    merged.loc[mask, "calc_ddG_err"] = merged.loc[mask, "bidir_err"]
    merged.loc[mask, "bidir_replaced"] = True
    merged.loc[mask, "bidir_mode"] = "average"
    merged.loc[mask, "bidir_observation"] = "average"
    return merged.drop(columns=["bidir_avg", "bidir_err"])

def expand_bidirectional_edges(edge_df: pd.DataFrame, bidir_df: pd.DataFrame) -> pd.DataFrame:
    """Use forward/reverse charge calculations as separate edge observations.

    Charge edges with paired reverse FEPs are removed from ``edge_df`` and
    replaced by two rows:
      from->to with ddG_f, and to->from with ddG_r.
    Non-charge edges are left unchanged.
    """
    out = edge_df.copy()
    out["bidir_replaced"] = False
    out["bidir_mode"] = "none"
    out["bidir_observation"] = "none"
    out["bidir_pair_id"] = ""
    out["bidir_hysteresis"] = np.nan
    if bidir_df.empty:
        return out

    key_cols = ["source", "system", "from_mutation", "to_mutation"]
    forward_keys = set()
    reverse_keys = set()
    replacement_rows = []

    indexed = {
        tuple(row[col] for col in key_cols): i
        for i, row in edge_df.reset_index(drop=True).iterrows()
    }

    for bidir in bidir_df.itertuples(index=False):
        source = getattr(bidir, "source")
        system = getattr(bidir, "system")
        src = getattr(bidir, "from_mutation")
        dst = getattr(bidir, "to_mutation")
        fwd_key = (source, system, src, dst)
        rev_key = (source, system, dst, src)
        if fwd_key not in indexed:
            continue

        forward_keys.add(fwd_key)
        reverse_keys.add(rev_key)
        template = edge_df.reset_index(drop=True).iloc[indexed[fwd_key]].copy()
        pair_id = f"{source}|{system}|{src}->{dst}"

        # crystal-unidirectional rule: when direction_mode is 'fwd'/'rev', emit
        # only that single crystal-seeded observation instead of both legs.
        mode = str(getattr(bidir, "direction_mode", "bidir") or "bidir")

        fwd = template.copy()
        fwd["from_mutation"] = src
        fwd["to_mutation"] = dst
        fwd["calc_ddG"] = float(getattr(bidir, "ddG_f"))
        fwd["calc_ddG_err"] = float(getattr(bidir, "ddG_f_err"))
        fwd["bidir_replaced"] = True
        fwd["bidir_mode"] = "uni" if mode == "fwd" else "multi"
        fwd["bidir_observation"] = "forward"
        fwd["bidir_pair_id"] = pair_id
        fwd["bidir_hysteresis"] = float(getattr(bidir, "hysteresis"))

        rev = template.copy()
        rev["from_mutation"] = dst
        rev["to_mutation"] = src
        rev["calc_ddG"] = float(getattr(bidir, "ddG_r"))
        rev["calc_ddG_err"] = float(getattr(bidir, "ddG_r_err"))
        rev["bidir_replaced"] = True
        rev["bidir_mode"] = "uni" if mode == "rev" else "multi"
        rev["bidir_observation"] = "reverse"
        rev["bidir_pair_id"] = pair_id
        rev["bidir_hysteresis"] = float(getattr(bidir, "hysteresis"))

        if mode == "fwd":
            replacement_rows.append(fwd)
        elif mode == "rev":
            replacement_rows.append(rev)
        else:
            replacement_rows.extend([fwd, rev])

    if not replacement_rows:
        return out

    def is_replaced(row: pd.Series) -> bool:
        key = tuple(row[col] for col in key_cols)
        return key in forward_keys or key in reverse_keys

    keep = out.loc[~out.apply(is_replaced, axis=1)].copy()
    repl = pd.DataFrame(replacement_rows, columns=out.columns)
    return pd.concat([keep, repl], ignore_index=True)

def _merge_cycle_scores(
    edge_group: pd.DataFrame,
    *,
    max_alt_path_len: int,
    max_paths_per_edge: int,
    min_sigma: float,
    use_bar_error: bool,
    bar_free_scale: float,
    cycle_weight_alpha: float,
) -> pd.DataFrame:
    scores = cycle_residual_edge_scores(
        edge_group,
        err_col="calc_ddG_err" if use_bar_error else None,
        max_path_len=max_alt_path_len,
        max_paths_per_edge=max_paths_per_edge,
        min_sigma=min_sigma,
        default_sigma=bar_free_scale,
        weight_alpha=cycle_weight_alpha,
    )
    if scores.empty:
        out = edge_group.copy()
        out["cycle_alt_n_paths"] = 0
        out["cycle_alt_median_abs_residual"] = 0.0
        out["cycle_q"] = 0.0
        out["cycle_weight_factor"] = 1.0
        return out

    out = edge_group.merge(scores, on=["from_mutation", "to_mutation"], how="left")
    out["cycle_alt_n_paths"] = out["cycle_alt_n_paths"].fillna(0)
    out["cycle_alt_median_abs_residual"] = out["cycle_alt_median_abs_residual"].fillna(0.0)
    out["cycle_q"] = out["cycle_q"].fillna(0.0)
    out["cycle_weight_factor"] = out["cycle_weight_factor"].fillna(1.0)
    return out

def fit_method(
    edge_df: pd.DataFrame,
    *,
    method_name: str,
    fit_kind: str,
    use_cycle_weights: bool,
    huber_delta: float,
    max_alt_path_len: int,
    max_paths_per_edge: int,
    min_sigma: float,
    use_bar_error: bool = True,
    bar_free_scale: float = 1.0,
    cycle_weight_alpha: float = 1.0,
    use_bar_error_for_cycle: bool = True,
    max_iter: int = 50,
    l1_eps: float = 1e-3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    node_pieces = []
    edge_pieces = []
    for (source, system), group in edge_df.groupby(["source", "system"], dropna=False):
        work = group.dropna(subset=["calc_ddG"]).copy()
        if work.empty:
            continue
        if use_cycle_weights:
            work = _merge_cycle_scores(
                work,
                max_alt_path_len=max_alt_path_len,
                max_paths_per_edge=max_paths_per_edge,
                min_sigma=min_sigma,
                use_bar_error=use_bar_error_for_cycle,
                bar_free_scale=bar_free_scale,
                cycle_weight_alpha=cycle_weight_alpha,
            )
            edge_weight_col = "cycle_weight_factor"
        else:
            edge_weight_col = None

        fit = fit_node_potentials(
            work,
            method=fit_kind,
            huber_delta=huber_delta,
            min_sigma=min_sigma,
            err_col="calc_ddG_err" if use_bar_error else None,
            default_sigma=None if use_bar_error else bar_free_scale,
            edge_weight_col=edge_weight_col,
            max_iter=max_iter,
            l1_eps=l1_eps,
        )
        if not fit.nodes.empty:
            nodes = fit.nodes.copy()
            nodes.insert(0, "method", method_name)
            nodes.insert(0, "system", system)
            nodes.insert(0, "source", source)
            nodes["converged"] = fit.converged
            nodes["n_iter"] = fit.n_iter
            node_pieces.append(nodes)
        if not fit.edges.empty:
            edges = fit.edges.copy()
            edges.insert(0, "method", method_name)
            edge_pieces.append(edges)

    node_df = pd.concat(node_pieces, ignore_index=True) if node_pieces else pd.DataFrame()
    edge_out = pd.concat(edge_pieces, ignore_index=True) if edge_pieces else pd.DataFrame()
    return node_df, edge_out

def run_estimators(
    edge_df: pd.DataFrame,
    bidir_df: pd.DataFrame,
    *,
    huber_delta: float,
    bar_free_scale: float,
    max_alt_path_len: int,
    max_paths_per_edge: int,
    min_sigma: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    bidir_multi_edges = expand_bidirectional_edges(edge_df, bidir_df)
    configs = [
        ("node_bidir_multi_wls_unit", bidir_multi_edges, "wls", False, 0.0),
        ("node_bidir_multi_huber_unit", bidir_multi_edges, "huber", False, 0.0),
        ("node_bidir_multi_cycle_huber_a010", bidir_multi_edges, "huber", True, 0.10),
        ("node_bidir_multi_cycle_huber_a025", bidir_multi_edges, "huber", True, 0.25),
        ("node_bidir_multi_cycle_huber_a100", bidir_multi_edges, "huber", True, 1.00),
    ]
    node_pieces = []
    edge_pieces = []
    for method_name, edges, fit_kind, use_cycle, cycle_alpha in configs:
        nodes, edge_res = fit_method(
            edges,
            method_name=method_name,
            fit_kind=fit_kind,
            use_cycle_weights=use_cycle,
            huber_delta=huber_delta,
            max_alt_path_len=max_alt_path_len,
            max_paths_per_edge=max_paths_per_edge,
            min_sigma=min_sigma,
            use_bar_error=False,
            bar_free_scale=bar_free_scale,
            cycle_weight_alpha=cycle_alpha,
            use_bar_error_for_cycle=False,
        )
        node_pieces.append(nodes)
        edge_pieces.append(edge_res)
    return pd.concat(node_pieces, ignore_index=True), pd.concat(edge_pieces, ignore_index=True)

def _wcc_nodes_from_edges(
    edge_group: pd.DataFrame,
    *,
    source: str,
    system: str,
    method_name: str,
    uncertainty_col: str,
) -> pd.DataFrame:
    cols = ["from_mutation", "to_mutation", "calc_ddG", uncertainty_col]
    result = wcc_from_dataframe(
        edge_group[cols],
        ref_mol="WT",
        mol1_col="from_mutation",
        mol2_col="to_mutation",
        ddg_col="calc_ddG",
        uncertainty_col=uncertainty_col,
    )
    if result is None:
        return pd.DataFrame()

    # wcc_from_dataframe keeps channel 0 as unweighted and channel 1 as the
    # first weighted channel. For the cycle-weighted variants we must read
    # channel 1; otherwise the supplied weights are silently ignored.
    energy_channel = 1 if len(result["energies"]) > 1 else 0
    return pd.DataFrame(
        {
            "source": source,
            "system": system,
            "method": method_name,
            "node": result["molecules"],
            "energy": result["energies"][energy_channel],
            "energy_err": result["path_dependent_errors"],
            "converged": True,
            "n_iter": np.nan,
        }
    )

def _reference_component_edges(edge_group: pd.DataFrame, ref_node: str = "WT") -> pd.DataFrame:
    graph = nx.Graph()
    graph.add_edges_from(zip(edge_group["from_mutation"].astype(str), edge_group["to_mutation"].astype(str)))
    if ref_node not in graph:
        return edge_group.iloc[0:0].copy()
    keep_nodes = nx.node_connected_component(graph, ref_node)
    mask = edge_group["from_mutation"].astype(str).isin(keep_nodes) & edge_group["to_mutation"].astype(str).isin(keep_nodes)
    return edge_group.loc[mask].copy()

def run_cycle_weighted_wcc_estimators(
    edge_df: pd.DataFrame,
    bidir_df: pd.DataFrame,
    *,
    bar_free_scale: float,
    max_alt_path_len: int,
    max_paths_per_edge: int,
    min_sigma: float,
    cycle_weight_alpha: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run WCC with cycle weights converted to WCC-compatible uncertainty.

    Existing WCC interprets the supplied uncertainty column as a standard
    deviation, then squares it internally and distributes cycle residuals in
    proportion to that variance. Therefore a cycle confidence factor ``f`` must be
    converted as:

        pseudo_var = c0^2 / f
        sigma_eff = sqrt(pseudo_var)

    The ``pseudo_var`` and ``sigma_eff`` method names are both emitted, but the
    values are equivalent when used correctly with the current WCC API.
    """
    simple_edges = apply_bidirectional_edges(edge_df, bidir_df)
    node_pieces = []
    edge_pieces = []
    for (source, system), group in simple_edges.groupby(["source", "system"], dropna=False):
        work = group.dropna(subset=["calc_ddG"]).copy()
        if work.empty:
            continue
        work = _reference_component_edges(work, ref_node="WT")
        if work.empty:
            continue
        work = _merge_cycle_scores(
            work,
            max_alt_path_len=max_alt_path_len,
            max_paths_per_edge=max_paths_per_edge,
            min_sigma=min_sigma,
            use_bar_error=False,
            bar_free_scale=bar_free_scale,
            cycle_weight_alpha=cycle_weight_alpha,
        )
        factor = pd.to_numeric(work["cycle_weight_factor"], errors="coerce").to_numpy(dtype=float)
        factor = np.where(np.isfinite(factor) & (factor > 0), factor, 1.0)
        work["wcc_cycle_pseudo_var"] = (float(bar_free_scale) ** 2) / factor
        work["wcc_cycle_sigma_eff"] = np.sqrt(work["wcc_cycle_pseudo_var"])
        # This column is intentionally identical to sigma_eff. It documents the
        # correct way to pass a precomputed pseudo-variance through the current
        # WCC API, which expects a sigma-like column and squares it internally.
        work["wcc_cycle_sigma_from_pseudo_var"] = np.sqrt(work["wcc_cycle_pseudo_var"])
        work["wcc_cycle_alpha"] = cycle_weight_alpha

        for method_name, uncertainty_col in [
            ("wcc_bidir_cycle_sigma_eff_a100", "wcc_cycle_sigma_eff"),
            ("wcc_bidir_cycle_pseudo_var_a100", "wcc_cycle_sigma_from_pseudo_var"),
        ]:
            nodes = _wcc_nodes_from_edges(
                work,
                source=source,
                system=system,
                method_name=method_name,
                uncertainty_col=uncertainty_col,
            )
            if not nodes.empty:
                node_pieces.append(nodes)

        edge_pieces.append(work)

    node_df = pd.concat(node_pieces, ignore_index=True) if node_pieces else pd.DataFrame()
    edge_out = pd.concat(edge_pieces, ignore_index=True) if edge_pieces else pd.DataFrame()
    return node_df, edge_out

def add_predictions(base_df: pd.DataFrame, node_predictions: pd.DataFrame) -> pd.DataFrame:
    if node_predictions.empty:
        return base_df.copy()
    value_wide = (
        node_predictions.pivot_table(
            index=["source", "system", "node"], columns="method", values="energy", aggfunc="first"
        )
        .reset_index()
        .rename(columns={"node": "mutation"})
    )
    err_wide = (
        node_predictions.pivot_table(
            index=["source", "system", "node"], columns="method", values="energy_err", aggfunc="first"
        )
        .add_suffix("_err")
        .reset_index()
        .rename(columns={"node": "mutation"})
    )
    out = base_df.merge(value_wide, on=["source", "system", "mutation"], how="left")
    out = out.merge(err_wide, on=["source", "system", "mutation"], how="left")
    return out

def metric_row(df: pd.DataFrame, method: str, scope: str, label: str) -> dict:
    sub = df[["exp_ddG", method, "system"]].dropna()
    if sub.empty:
        return {
            "scope": scope,
            "method": method,
            "label": label,
            "n": 0,
            "n_systems": 0,
            "rmse": np.nan,
            "mae": np.nan,
            "r2": np.nan,
            "median_abs_error": np.nan,
        }
    y = sub["exp_ddG"].to_numpy(dtype=float)
    pred = sub[method].to_numpy(dtype=float)
    resid = pred - y
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = np.nan if ss_tot == 0 else 1.0 - ss_res / ss_tot
    return {
        "scope": scope,
        "method": method,
        "label": label,
        "n": int(len(sub)),
        "n_systems": int(sub["system"].nunique()),
        "rmse": float(np.sqrt(np.mean(resid ** 2))),
        "mae": float(np.mean(np.abs(resid))),
        "r2": r2,
        "median_abs_error": float(np.median(np.abs(resid))),
    }

def compute_metrics(target_df: pd.DataFrame, all_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, label in PLOT_METHODS:
        for scope, df in [("Target", target_df), ("All", all_df)]:
            if method in df.columns:
                rows.append(metric_row(df, method, scope, label))
    return pd.DataFrame(rows)

def _alpha_method_name(alpha: float) -> str:
    return f"cwne_alpha_{alpha:g}".replace(".", "p")

def run_cycle_strength_sweep(
    edge_df: pd.DataFrame,
    bidir_df: pd.DataFrame,
    target_df: pd.DataFrame,
    all_df: pd.DataFrame,
    *,
    huber_delta: float,
    bar_free_scale: float,
    max_alt_path_len: int,
    max_paths_per_edge: int,
    min_sigma: float,
) -> pd.DataFrame:
    bidir_multi_edges = expand_bidirectional_edges(edge_df, bidir_df)
    rows = []
    for alpha in CYCLE_SWEEP_ALPHAS:
        method = _alpha_method_name(alpha)
        nodes, _ = fit_method(
            bidir_multi_edges,
            method_name=method,
            fit_kind="huber",
            use_cycle_weights=alpha > 0,
            huber_delta=huber_delta,
            max_alt_path_len=max_alt_path_len,
            max_paths_per_edge=max_paths_per_edge,
            min_sigma=min_sigma,
            use_bar_error=False,
            bar_free_scale=bar_free_scale,
            cycle_weight_alpha=alpha,
            use_bar_error_for_cycle=False,
        )
        target_aug = add_predictions(target_df, nodes)
        all_aug = add_predictions(all_df, nodes)
        for scope, df in [("Target", target_aug), ("All", all_aug)]:
            metric = metric_row(df, method, scope, f"alpha={alpha:g}")
            metric["cycle_alpha"] = alpha
            rows.append(metric)
    return pd.DataFrame(rows)

def _method_values(df: pd.DataFrame, method: str) -> tuple[np.ndarray, np.ndarray]:
    sub = df[["exp_ddG", method]].dropna()
    return sub["exp_ddG"].to_numpy(dtype=float), sub[method].to_numpy(dtype=float)

def save_figure(fig: plt.Figure, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

def plot_abs_error_boxplot(df: pd.DataFrame, title: str, out_base: Path) -> None:
    rng = np.random.default_rng(0)
    data = []
    labels = []
    colors = []
    palette = ["#e8ad36", "#8dbf67", "#5aa0c8", "#7e5aa7", "#4c78a8", "#6f9d4e", "#55a868", "#c44e52", "#8172b3", "#937860"]
    for i, (method, label) in enumerate(PLOT_METHODS):
        if method not in df.columns:
            continue
        y, pred = _method_values(df, method)
        if len(y) == 0:
            continue
        data.append(np.abs(pred - y))
        labels.append(label)
        colors.append(palette[i % len(palette)])

    fig, ax = plt.subplots(figsize=(max(9, len(data) * 1.2), 4.8))
    box = ax.boxplot(data, patch_artist=True, widths=0.58, showfliers=False)
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)
    for i, values in enumerate(data, start=1):
        x = rng.normal(i, 0.045, size=len(values))
        ax.scatter(x, values, s=22, color=colors[i - 1], edgecolor="white", linewidth=0.4, alpha=0.9)
    ax.axhline(1.0, color="0.55", linestyle="--", linewidth=1.0)
    ax.axhline(2.0, color="0.65", linestyle=":", linewidth=1.0)
    ax.set_title(title)
    ax.set_ylabel("|FEP - Exp| (kcal/mol)")
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.18)
    save_figure(fig, out_base)

def plot_metrics_bar(metrics: pd.DataFrame, out_base: Path) -> None:
    target = metrics[(metrics["scope"] == "Target") & (metrics["n"] > 0)].copy()
    order = [method for method, _ in PLOT_METHODS if method in set(target["method"])]
    target["method"] = pd.Categorical(target["method"], order, ordered=True)
    target = target.sort_values("method")

    x = np.arange(len(target))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(9, len(target) * 1.0), 4.6))
    ax.bar(x - width / 2, target["mae"], width, label="MAE", color="#4c78a8")
    ax.bar(x + width / 2, target["rmse"], width, label="RMSE", color="#f58518")
    ax.set_xticks(x)
    ax.set_xticklabels(target["label"], rotation=35, ha="right")
    ax.set_ylabel("kcal/mol")
    ax.set_title("Target rows: error metrics")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.18)
    save_figure(fig, out_base)

def plot_cycle_strength_sweep(sweep: pd.DataFrame, out_base: Path) -> None:
    target = sweep[sweep["scope"].eq("Target")].sort_values("cycle_alpha")
    if target.empty:
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(target["cycle_alpha"], target["mae"], marker="o", label="MAE", color="#4c78a8")
    ax.plot(target["cycle_alpha"], target["rmse"], marker="s", label="RMSE", color="#f58518")
    ax.set_xscale("symlog", linthresh=0.02)
    ax.set_xlabel("cycle weight alpha")
    ax.set_ylabel("kcal/mol")
    ax.set_title("Cycle-weight strength sweep")
    ax.grid(alpha=0.18)
    ax.legend(frameon=False)
    save_figure(fig, out_base)

def win_loss_table(
    df: pd.DataFrame,
    *,
    method_a: str,
    method_b: str,
    label_a: str,
    label_b: str,
    scope: str,
    tol: float = 1e-9,
) -> tuple[pd.DataFrame, dict]:
    sub = df[["source", "system", "mutation", "node_type", "exp_ddG", method_a, method_b]].dropna().copy()
    sub[f"{method_a}_abs_err"] = (sub[method_a] - sub["exp_ddG"]).abs()
    sub[f"{method_b}_abs_err"] = (sub[method_b] - sub["exp_ddG"]).abs()
    sub["abs_err_delta"] = sub[f"{method_a}_abs_err"] - sub[f"{method_b}_abs_err"]
    sub["winner"] = np.where(
        sub["abs_err_delta"] > tol,
        label_b,
        np.where(sub["abs_err_delta"] < -tol, label_a, "tie"),
    )
    summary = {
        "scope": scope,
        "method_a": method_a,
        "method_b": method_b,
        "label_a": label_a,
        "label_b": label_b,
        "n": int(len(sub)),
        "wins_a": int((sub["winner"] == label_a).sum()),
        "wins_b": int((sub["winner"] == label_b).sum()),
        "ties": int((sub["winner"] == "tie").sum()),
        "mean_delta_abs_error_a_minus_b": float(sub["abs_err_delta"].mean()) if len(sub) else np.nan,
        "median_delta_abs_error_a_minus_b": float(sub["abs_err_delta"].median()) if len(sub) else np.nan,
    }
    return sub, summary

def plot_win_loss_scatter(compare_df: pd.DataFrame, out_base: Path, *, title: str, label_a: str, label_b: str) -> None:
    if compare_df.empty:
        return
    fig, ax = plt.subplots(figsize=(5.8, 5.4))
    colors = {label_a: "#f58518", label_b: "#4c78a8", "tie": "0.55"}
    for winner, group in compare_df.groupby("winner"):
        ax.scatter(
            group[f"{COMPARE_A}_abs_err"],
            group[f"{COMPARE_B}_abs_err"],
            label=winner,
            s=42,
            color=colors.get(winner, "0.4"),
            edgecolor="white",
            linewidth=0.5,
            alpha=0.9,
        )
    max_val = float(np.ceil(max(compare_df[f"{COMPARE_A}_abs_err"].max(), compare_df[f"{COMPARE_B}_abs_err"].max()) + 0.25))
    ax.plot([0, max_val], [0, max_val], color="0.55", linestyle="--", linewidth=1.0)
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.set_xlabel(f"{label_a} absolute error")
    ax.set_ylabel(f"{label_b} absolute error")
    ax.set_title(title)
    ax.grid(alpha=0.16)
    ax.legend(frameon=False)
    save_figure(fig, out_base)

def plot_win_loss_delta(compare_df: pd.DataFrame, out_base: Path, *, title: str, label_b: str) -> None:
    if compare_df.empty:
        return
    plot_df = compare_df.sort_values("abs_err_delta", ascending=False).reset_index(drop=True)
    labels = [f"{row.system} {row.mutation}" for row in plot_df.itertuples(index=False)]
    colors = np.where(plot_df["abs_err_delta"] > 0, "#4c78a8", "#f58518")
    fig, ax = plt.subplots(figsize=(8.5, max(4.2, 0.28 * len(plot_df))))
    y = np.arange(len(plot_df))
    ax.barh(y, plot_df["abs_err_delta"], color=colors, alpha=0.82)
    ax.axvline(0, color="0.35", linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(f"abs error improvement of {label_b} over WCC-bidir (kcal/mol)")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.16)
    save_figure(fig, out_base)

def plot_scatter_grid(df: pd.DataFrame, out_base: Path) -> None:
    methods = ROBUST_METHODS
    valid_values = []
    for method, _ in methods:
        if method in df.columns:
            y, pred = _method_values(df, method)
            if len(y):
                valid_values.extend(y.tolist())
                valid_values.extend(pred.tolist())
    if not valid_values:
        return
    lim_min = float(np.floor(min(valid_values) - 0.5))
    lim_max = float(np.ceil(max(valid_values) + 0.5))

    ncols = 3
    nrows = int(np.ceil(len(methods) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10.5, 3.25 * nrows), sharex=True, sharey=True)
    axes = np.ravel(axes)
    for ax, (method, label) in zip(axes, methods):
        y, pred = _method_values(df, method)
        ax.axline((0, 0), slope=1, color="0.6", linestyle="--", linewidth=1.0)
        if len(y):
            resid = pred - y
            mae = np.mean(np.abs(resid))
            ax.scatter(y, pred, s=34, color="#4c78a8", edgecolor="white", linewidth=0.5)
            ax.text(0.03, 0.97, f"MAE={mae:.2f}", transform=ax.transAxes, va="top")
        ax.set_title(label)
        ax.set_xlim(lim_min, lim_max)
        ax.set_ylim(lim_min, lim_max)
        ax.grid(alpha=0.16)
    for ax in axes[3:]:
        ax.set_xlabel("Experiment ddG (kcal/mol)")
    for ax in axes[::3]:
        ax.set_ylabel("Prediction ddG (kcal/mol)")
    for ax in axes[len(methods):]:
        ax.axis("off")
    fig.suptitle("Robust graph estimators on target rows", y=1.02)
    save_figure(fig, out_base)

def plot_cycle_qc(edge_residuals: pd.DataFrame, out_base: Path) -> None:
    method = "node_bidir_multi_cycle_huber_a025"
    sub = edge_residuals[edge_residuals["method"] == method].copy()
    if sub.empty or "cycle_q" not in sub.columns:
        return
    sub["abs_residual"] = sub["node_fit_residual"].abs()
    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    ax.scatter(
        sub["cycle_q"],
        sub["abs_residual"],
        s=28,
        c=sub["cycle_weight_factor"],
        cmap="viridis",
        edgecolor="white",
        linewidth=0.4,
    )
    ax.set_xlabel("Alternative-path cycle q")
    ax.set_ylabel("|node-fit edge residual| (kcal/mol)")
    ax.set_title("Cycle reweighting diagnostic")
    cb = fig.colorbar(ax.collections[0], ax=ax)
    cb.set_label("edge weight factor")
    ax.grid(alpha=0.16)
    save_figure(fig, out_base)

def write_readme(output_dir: Path, input_dir: Path, metrics: pd.DataFrame) -> None:
    target = metrics[(metrics["scope"] == "Target") & (metrics["n"] > 0)].copy()
    target = target.sort_values(["mae", "rmse"])
    win_loss_path = output_dir / "win_loss_summary_wcc_bidir_vs_cwne.csv"
    win_loss = pd.read_csv(win_loss_path) if win_loss_path.exists() else pd.DataFrame()
    lines = [
        "# Robust graph estimator outputs",
        "",
        f"Input: `{input_dir}`",
        "",
        "This directory was generated without modifying the original result directory.",
        "",
        "New estimator columns:",
        "- `wcc_bidir_cycle_sigma_eff_a100`: WCC on bidirectional averaged edges, using cycle residual weights converted to effective sigma.",
        "- `wcc_bidir_cycle_pseudo_var_a100`: same WCC weights expressed as pseudo-variance and passed through the current WCC sigma API.",
        "- `node_bidir_multi_wls_unit`: bidirectional forward/reverse observations with unit edge scale.",
        "- `node_bidir_multi_huber_unit`: Huber node-potential fit with unit edge scale.",
        "- `node_bidir_multi_cycle_huber_a100` (**CWNE**): Huber node-potential fit with bidirectional multi-edge observations and fixed alpha=1 cycle downweights.",
        "- `node_bidir_multi_cycle_huber_a010/a025`: exploratory CWNE cycle-strength variants.",
        "",
        "The figures intentionally omit estimator variants that use BAR errors as weights or Huber scales.",
        "",
        "Target-scope metrics sorted by MAE:",
        "",
        "| method | n | RMSE | MAE | R2 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in target.itertuples(index=False):
        lines.append(f"| {row.method} | {row.n} | {row.rmse:.3f} | {row.mae:.3f} | {row.r2:.3f} |")
    if not win_loss.empty:
        lines.extend(
            [
                "",
                "Win/loss versus WCC-bidir:",
                "",
                "| scope | n | WCC-bidir wins | CWNE wins | ties | mean abs-error delta |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in win_loss.itertuples(index=False):
            lines.append(
                f"| {row.scope} | {row.n} | {row.wins_a} | {row.wins_b} | {row.ties} | "
                f"{row.mean_delta_abs_error_a_minus_b:.3f} |"
            )
    lines.append("")
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")

def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures" / "png").mkdir(parents=True, exist_ok=True)
    (output_dir / "figures" / "pdf").mkdir(parents=True, exist_ok=True)

    target_df, all_df, edge_df, bidir_df = read_inputs(input_dir)
    node_predictions, edge_residuals = run_estimators(
        edge_df,
        bidir_df,
        huber_delta=args.huber_delta,
        bar_free_scale=args.bar_free_scale,
        max_alt_path_len=args.max_alt_path_len,
        max_paths_per_edge=args.max_paths_per_edge,
        min_sigma=args.min_sigma,
    )
    wcc_nodes, wcc_weight_edges = run_cycle_weighted_wcc_estimators(
        edge_df,
        bidir_df,
        bar_free_scale=args.bar_free_scale,
        max_alt_path_len=args.max_alt_path_len,
        max_paths_per_edge=args.max_paths_per_edge,
        min_sigma=args.min_sigma,
        cycle_weight_alpha=1.0,
    )
    if not wcc_nodes.empty:
        node_predictions = pd.concat([node_predictions, wcc_nodes], ignore_index=True)

    target_aug = add_predictions(target_df, node_predictions)
    all_aug = add_predictions(all_df, node_predictions)
    metrics = compute_metrics(target_aug, all_aug)
    cycle_sweep = run_cycle_strength_sweep(
        edge_df,
        bidir_df,
        target_df,
        all_df,
        huber_delta=args.huber_delta,
        bar_free_scale=args.bar_free_scale,
        max_alt_path_len=args.max_alt_path_len,
        max_paths_per_edge=args.max_paths_per_edge,
        min_sigma=args.min_sigma,
    )
    target_compare, target_summary = win_loss_table(
        target_aug,
        method_a=COMPARE_A,
        method_b=COMPARE_B,
        label_a=COMPARE_A_LABEL,
        label_b=COMPARE_B_LABEL,
        scope="Target",
    )
    all_compare, all_summary = win_loss_table(
        all_aug,
        method_a=COMPARE_A,
        method_b=COMPARE_B,
        label_a=COMPARE_A_LABEL,
        label_b=COMPARE_B_LABEL,
        scope="All",
    )
    win_loss_summary = pd.DataFrame([target_summary, all_summary])

    target_aug.to_csv(output_dir / "target_results_robust_graph.csv", index=False)
    all_aug.to_csv(output_dir / "all_results_robust_graph.csv", index=False)
    node_predictions.to_csv(output_dir / "node_potentials.csv", index=False)
    edge_residuals.to_csv(output_dir / "edge_residuals.csv", index=False)
    wcc_weight_edges.to_csv(output_dir / "wcc_cycle_weight_edges.csv", index=False)
    metrics.to_csv(output_dir / "metrics_robust_graph.csv", index=False)
    cycle_sweep.to_csv(output_dir / "cycle_strength_sweep.csv", index=False)
    target_compare.to_csv(output_dir / "win_loss_target_wcc_bidir_vs_cwne.csv", index=False)
    all_compare.to_csv(output_dir / "win_loss_all_wcc_bidir_vs_cwne.csv", index=False)
    win_loss_summary.to_csv(output_dir / "win_loss_summary_wcc_bidir_vs_cwne.csv", index=False)

    plot_abs_error_boxplot(
        target_aug,
        "Absolute error - target rows",
        output_dir / "figures" / "png" / "target_abs_error_boxplot",
    )
    plot_abs_error_boxplot(
        all_aug,
        "Absolute error - all data points (targets + intermediates)",
        output_dir / "figures" / "png" / "all_abs_error_boxplot",
    )
    plot_metrics_bar(metrics, output_dir / "figures" / "png" / "target_metrics_bar")
    plot_scatter_grid(target_aug, output_dir / "figures" / "png" / "target_robust_scatter")
    plot_cycle_qc(edge_residuals, output_dir / "figures" / "png" / "cycle_reweighting_diagnostic")
    plot_cycle_strength_sweep(cycle_sweep, output_dir / "figures" / "png" / "cycle_strength_sweep")
    plot_win_loss_scatter(
        target_compare,
        output_dir / "figures" / "png" / "target_win_loss_abs_error_scatter",
        title="Target rows: WCC-bidir vs CWNE",
        label_a=COMPARE_A_LABEL,
        label_b=COMPARE_B_LABEL,
    )
    plot_win_loss_scatter(
        all_compare,
        output_dir / "figures" / "png" / "all_win_loss_abs_error_scatter",
        title="All rows: WCC-bidir vs CWNE",
        label_a=COMPARE_A_LABEL,
        label_b=COMPARE_B_LABEL,
    )
    plot_win_loss_delta(
        target_compare,
        output_dir / "figures" / "png" / "target_win_loss_delta",
        title="Target rows: row-level absolute-error change",
        label_b=COMPARE_B_LABEL,
    )
    plot_win_loss_delta(
        all_compare,
        output_dir / "figures" / "png" / "all_win_loss_delta",
        title="All rows: row-level absolute-error change",
        label_b=COMPARE_B_LABEL,
    )

    # Mirror PDFs under figures/pdf for easier browsing.
    for pdf in (output_dir / "figures" / "png").glob("*.pdf"):
        pdf.replace(output_dir / "figures" / "pdf" / pdf.name)

    write_readme(output_dir, input_dir, metrics)
    print(f"Wrote robust graph outputs to {output_dir}")
    print(metrics[metrics["scope"] == "Target"].sort_values("mae").to_string(index=False))

if __name__ == "__main__":
    main()
