#!/usr/bin/env python
"""Fit node potentials and compare prediction errors."""
from __future__ import annotations

import functools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import protmutmap.robust_graph as rg
import scripts.analyze_robust_graph_estimators as age

RES = ROOT / "results/source_data/fep"
OUT = ROOT / "results/source_data/huber"
ROSETTA = OUT / "rosetta_flex_ddg.csv"
NBOOT = 20_000
SEED = 20260824
R_CLIP = 0.9999

EDGES = age.expand_bidirectional_edges(
    pd.read_csv(RES / "edge_diagnostics.csv"),
    pd.read_csv(RES / "edge_bidirectional.csv"),
)
_ORIG_CYCLE_SCORE = rg.cycle_residual_edge_scores

def mutation_key(value: str) -> str:
    return ",".join(sorted(str(value).replace("+", ",").split(",")))

def fit_nodes(*, kind: str = "huber", cycle: bool = False) -> pd.Series:
    age.cycle_residual_edge_scores = functools.partial(
        _ORIG_CYCLE_SCORE, min_weight_factor=0.05
    )
    try:
        nodes, _ = age.fit_method(
            EDGES,
            method_name=kind,
            fit_kind=kind,
            use_cycle_weights=cycle,
            huber_delta=1.5,
            max_alt_path_len=6,
            max_paths_per_edge=64,
            min_sigma=0.05,
            use_bar_error=False,
            bar_free_scale=1.0,
            cycle_weight_alpha=1.0,
            use_bar_error_for_cycle=False,
        )
    finally:
        age.cycle_residual_edge_scores = _ORIG_CYCLE_SCORE
    return nodes.set_index(["source", "system", "node"])["energy"]

def lookup_nodes(series: pd.Series, frame: pd.DataFrame, node_col: str = "mutation") -> np.ndarray:
    return np.array(
        [series.get((row.source, row.system, getattr(row, node_col)), np.nan) for row in frame.itertuples()],
        dtype=float,
    )

def build_predictions() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    all_results = pd.read_csv(RES / "all_results.csv")
    cols = [
        "source", "system", "mutation", "mutation_num", "node_type", "exp_ddG",
        "Additive", "Stepwise",
    ]
    nodes = all_results[cols].dropna(
        subset=["exp_ddG", "Additive", "Stepwise"]
    ).copy()
    nodes["mutation_key"] = nodes["mutation"].map(mutation_key)
    nodes["variant_id"] = nodes["system"] + "|" + nodes["mutation_key"]

    fits = {
        "OLS": fit_nodes(kind="wls", cycle=False),
        "ProtMutMap": fit_nodes(kind="huber", cycle=False),
    }
    for label, fit in fits.items():
        nodes[label] = lookup_nodes(fit, nodes)

    edge_source = pd.read_csv(RES / "edge_accuracy.csv")
    edges = edge_source[
        ["source", "system", "from_mutation", "to_mutation", "exp_edge_ddG", "approx_wt", "raw"]
    ].copy()
    edges = edges.rename(columns={"approx_wt": "Additive", "raw": "Raw_FEP"})
    for label, fit in fits.items():
        edges[label] = np.array(
            [
                fit.get((row.source, row.system, row.to_mutation), np.nan)
                - fit.get((row.source, row.system, row.from_mutation), np.nan)
                for row in edges.itertuples()
            ],
            dtype=float,
        )

    rosetta = pd.read_csv(ROSETTA).copy()
    rosetta["mutation_key"] = rosetta["mutation"].map(mutation_key)
    rosetta["variant_id"] = rosetta["system"] + "|" + rosetta["mutation_key"]
    ros_map = rosetta.set_index("variant_id")["ddg_gam"]
    nodes["Rosetta_flex_ddG"] = nodes["variant_id"].map(ros_map)

    method_cols = [
        "Additive", "Stepwise", "OLS", "ProtMutMap",
        "Rosetta_flex_ddG",
    ]
    grouping = {
        "exp_ddG": "first",
        "mutation_num": "first",
        "node_type": "first",
        "source": lambda x: ";".join(sorted(x)),
        **{name: "mean" for name in method_cols},
    }
    external_mean = nodes.groupby(
        ["variant_id", "system", "mutation_key"], as_index=False
    ).agg(grouping)
    external_mean["source_count"] = nodes.groupby("variant_id").size().reindex(
        external_mean["variant_id"]
    ).to_numpy()

    def preferred_source(prefix: str) -> pd.DataFrame:
        ordered = nodes.assign(
            _preferred=(~nodes["source"].str.startswith(prefix)).astype(int)
        ).sort_values(["variant_id", "_preferred", "source"])
        return ordered.drop_duplicates("variant_id", keep="first").drop(columns="_preferred")

    counts = nodes.groupby("variant_id").size()
    no_duplicates = nodes[nodes["variant_id"].map(counts).eq(1)].copy()
    external_scopes = {
        "external_unique28_mean": external_mean,
        "external_unique28_standard_preferred": preferred_source("standard"),
        "external_unique28_charge_preferred": preferred_source("charge"),
        "external_unique26_no_duplicates": no_duplicates,
    }
    return nodes, edges, external_scopes

def _pearson_rows(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    yc = y - y.mean(axis=1, keepdims=True)
    pc = p - p.mean(axis=1, keepdims=True)
    denominator = np.sqrt((yc * yc).sum(axis=1) * (pc * pc).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        value = (yc * pc).sum(axis=1) / denominator
    return np.where(denominator > 0, value, np.nan)

def _spearman_rows(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    return _pearson_rows(rankdata(y, axis=1), rankdata(p, axis=1))

def _point_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    residual = p - y
    pearson_r = float(np.corrcoef(y, p)[0, 1])
    spearman = float(_spearman_rows(y[None, :], p[None, :])[0])
    return {
        "rmse": float(np.sqrt(np.mean(residual * residual))),
        "mae": float(np.mean(np.abs(residual))),
        "median_abs": float(np.median(np.abs(residual))),
        "p90_abs": float(np.percentile(np.abs(residual), 90)),
        "pearson_r": pearson_r,
        "fisher_z": float(np.arctanh(np.clip(pearson_r, -R_CLIP, R_CLIP))),
        "spearman": spearman,
        "spearman_z": float(np.arctanh(np.clip(spearman, -R_CLIP, R_CLIP))),
    }

def _bootstrap_metrics(y: np.ndarray, p: np.ndarray, idx: np.ndarray) -> dict[str, np.ndarray]:
    yy, pp = y[idx], p[idx]
    residual = pp - yy
    pearson_r = _pearson_rows(yy, pp)
    spearman = _spearman_rows(yy, pp)
    return {
        "rmse": np.sqrt(np.mean(residual * residual, axis=1)),
        "mae": np.mean(np.abs(residual), axis=1),
        "median_abs": np.median(np.abs(residual), axis=1),
        "p90_abs": np.percentile(np.abs(residual), 90, axis=1),
        "pearson_r": pearson_r,
        "fisher_z": np.arctanh(np.clip(pearson_r, -R_CLIP, R_CLIP)),
        "spearman": spearman,
        "spearman_z": np.arctanh(np.clip(spearman, -R_CLIP, R_CLIP)),
    }

def _bca(theta: float, boot: np.ndarray, jack: np.ndarray) -> tuple[float, float]:
    boot = np.asarray(boot, dtype=float)
    boot = boot[np.isfinite(boot)]
    jack = np.asarray(jack, dtype=float)
    jack = jack[np.isfinite(jack)]
    if not np.isfinite(theta) or len(boot) < 100 or len(jack) < 3:
        return np.nan, np.nan
    proportion = np.mean(boot < theta)
    if proportion <= 0 or proportion >= 1:
        return np.nan, np.nan
    z0 = norm.ppf(proportion)
    jack_mean = jack.mean()
    numerator = np.sum((jack_mean - jack) ** 3)
    denominator = 6.0 * np.sum((jack_mean - jack) ** 2) ** 1.5
    if denominator == 0:
        return np.nan, np.nan
    acceleration = numerator / denominator

    def adjusted(z: float) -> float:
        den = 1.0 - acceleration * (z0 + z)
        return np.nan if den == 0 else norm.cdf(z0 + (z0 + z) / den)

    qlo = adjusted(norm.ppf(0.025))
    qhi = adjusted(norm.ppf(0.975))
    if not (np.isfinite(qlo) and np.isfinite(qhi)):
        return np.nan, np.nan
    qlo, qhi = np.clip([qlo, qhi], 0.0, 1.0)
    return float(np.percentile(boot, 100 * qlo)), float(np.percentile(boot, 100 * qhi))

def metric_row(scope: str, method: str, y: np.ndarray, p: np.ndarray, seed_offset: int) -> dict:
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    rng = np.random.default_rng(SEED + seed_offset)
    idx = rng.integers(0, len(y), (NBOOT, len(y)))
    point = _point_metrics(y, p)
    boot = _bootstrap_metrics(y, p, idx)
    row = {"scope": scope, "method": method, "n": len(y)}
    for name, theta in point.items():
        values = boot[name]
        finite = values[np.isfinite(values)]
        row[name] = theta
        row[f"{name}_lo"], row[f"{name}_hi"] = np.percentile(finite, [2.5, 97.5])
        jack = []
        for i in range(len(y)):
            jack.append(_point_metrics(np.delete(y, i), np.delete(p, i))[name])
        row[f"{name}_bca_lo"], row[f"{name}_bca_hi"] = _bca(theta, values, np.array(jack))
    return row

def paired_row(
    scope: str,
    reference: str,
    method: str,
    y: np.ndarray,
    p_ref: np.ndarray,
    p_other: np.ndarray,
    seed_offset: int,
    contrast_group: str = "vs_reference",
) -> dict:
    ok = np.isfinite(y) & np.isfinite(p_ref) & np.isfinite(p_other)
    y, p_ref, p_other = y[ok], p_ref[ok], p_other[ok]
    rng = np.random.default_rng(SEED + seed_offset)
    idx = rng.integers(0, len(y), (NBOOT, len(y)))

    ref_point = _point_metrics(y, p_ref)
    other_point = _point_metrics(y, p_other)
    ref_boot = _bootstrap_metrics(y, p_ref, idx)
    other_boot = _bootstrap_metrics(y, p_other, idx)
    error_names = ("rmse", "mae", "median_abs", "p90_abs")
    corr_names = ("pearson_r", "fisher_z", "spearman", "spearman_z")

    row = {
        "scope": scope,
        "contrast_group": contrast_group,
        "reference": reference,
        "method": method,
        "n": len(y),
        "error_difference": "method-reference",
        "correlation_difference": "reference-method",
    }
    for name in error_names + corr_names:
        if name in error_names:
            theta = other_point[name] - ref_point[name]
            values = other_boot[name] - ref_boot[name]
        else:
            theta = ref_point[name] - other_point[name]
            values = ref_boot[name] - other_boot[name]
        finite = values[np.isfinite(values)]
        row[f"d_{name}"] = theta
        row[f"d_{name}_lo"], row[f"d_{name}_hi"] = np.percentile(finite, [2.5, 97.5])
        jack = []
        for i in range(len(y)):
            yr, rr, oo = np.delete(y, i), np.delete(p_ref, i), np.delete(p_other, i)
            ref_j = _point_metrics(yr, rr)[name]
            other_j = _point_metrics(yr, oo)[name]
            jack.append(other_j - ref_j if name in error_names else ref_j - other_j)
        row[f"d_{name}_bca_lo"], row[f"d_{name}_bca_hi"] = _bca(
            theta, values, np.array(jack)
        )
    abs_ref = np.abs(p_ref - y)
    abs_other = np.abs(p_other - y)
    row["wins_reference"] = int(np.sum(abs_ref < abs_other))
    row["ties"] = int(np.sum(abs_ref == abs_other))
    return row

def main() -> None:
    OUT.mkdir(exist_ok=True)
    nodes, edges, external_scopes = build_predictions()
    nodes.to_csv(OUT / "node_predictions.csv", index=False)
    edges.to_csv(OUT / "edge_predictions.csv", index=False)
    external_scopes["external_unique28_mean"].to_csv(
        OUT / "external_unique28_predictions.csv", index=False
    )

    metrics_rows = []
    paired_rows = []
    seed_offset = 0

    node_methods = ["Additive", "Stepwise", "OLS", "ProtMutMap"]
    y_node = nodes["exp_ddG"].to_numpy(float)
    for method in node_methods:
        seed_offset += 1
        metrics_rows.append(metric_row(
            "internal_node30", method, y_node, nodes[method].to_numpy(float), seed_offset
        ))
    for method in node_methods:
        if method == "ProtMutMap":
            continue
        seed_offset += 1
        paired_rows.append(paired_row(
            "internal_node30", "ProtMutMap", method, y_node,
            nodes["ProtMutMap"].to_numpy(float), nodes[method].to_numpy(float), seed_offset,
            "ols_sensitivity" if method == "OLS" else "primary_vs_baseline",
        ))
    for method in ("Additive", "Stepwise"):
        seed_offset += 1
        paired_rows.append(paired_row(
            "internal_node30", "OLS", method, y_node,
            nodes["OLS"].to_numpy(float), nodes[method].to_numpy(float), seed_offset,
            "ols_sensitivity_vs_baseline",
        ))

    unique_internal = external_scopes["external_unique28_mean"]
    y_unique = unique_internal["exp_ddG"].to_numpy(float)
    for method in node_methods:
        seed_offset += 1
        metrics_rows.append(metric_row(
            "internal_unique28_mean_sensitivity", method, y_unique,
            unique_internal[method].to_numpy(float), seed_offset,
        ))
    for method in node_methods:
        if method == "ProtMutMap":
            continue
        seed_offset += 1
        paired_rows.append(paired_row(
            "internal_unique28_mean_sensitivity", "ProtMutMap", method, y_unique,
            unique_internal["ProtMutMap"].to_numpy(float),
            unique_internal[method].to_numpy(float), seed_offset,
            "ols_sensitivity" if method == "OLS" else "primary_vs_baseline",
        ))
    for method in ("Additive", "Stepwise"):
        seed_offset += 1
        paired_rows.append(paired_row(
            "internal_unique28_mean_sensitivity", "OLS", method, y_unique,
            unique_internal["OLS"].to_numpy(float),
            unique_internal[method].to_numpy(float), seed_offset,
            "ols_sensitivity_vs_baseline",
        ))

    edge_methods = ["Additive", "Raw_FEP", "OLS", "ProtMutMap"]
    y_edge = edges["exp_edge_ddG"].to_numpy(float)
    for method in edge_methods:
        seed_offset += 1
        metrics_rows.append(metric_row(
            "internal_edge20", method, y_edge, edges[method].to_numpy(float), seed_offset
        ))
    for method in edge_methods:
        if method == "ProtMutMap":
            continue
        seed_offset += 1
        paired_rows.append(paired_row(
            "internal_edge20", "ProtMutMap", method, y_edge,
            edges["ProtMutMap"].to_numpy(float), edges[method].to_numpy(float), seed_offset,
            "ols_sensitivity" if method == "OLS" else "diagnostic_vs_baseline",
        ))

    for scope, frame in external_scopes.items():
        y = frame["exp_ddG"].to_numpy(float)
        for method in ("ProtMutMap", "Rosetta_flex_ddG"):
            seed_offset += 1
            metrics_rows.append(metric_row(scope, method, y, frame[method].to_numpy(float), seed_offset))
        for method in ("Rosetta_flex_ddG",):
            seed_offset += 1
            paired_rows.append(paired_row(
                scope, "ProtMutMap", method, y,
                frame["ProtMutMap"].to_numpy(float), frame[method].to_numpy(float), seed_offset,
                "external_reference",
            ))

    metrics = pd.DataFrame(metrics_rows)
    paired = pd.DataFrame(paired_rows)
    metrics.to_csv(OUT / "method_metrics.csv", index=False)
    paired.to_csv(OUT / "paired_comparisons.csv", index=False)

    duplicates = nodes[nodes.duplicated("variant_id", keep=False)][
        ["source", "system", "mutation", "variant_id", "exp_ddG", "ProtMutMap", "Rosetta_flex_ddG"]
    ].sort_values(["variant_id", "source"])
    duplicates.to_csv(OUT / "external_duplicates.csv", index=False)

    display_cols = [
        "scope", "reference", "method", "n",
        "d_rmse", "d_rmse_bca_lo", "d_rmse_bca_hi",
        "d_mae", "d_mae_bca_lo", "d_mae_bca_hi",
        "d_fisher_z", "d_fisher_z_bca_lo", "d_fisher_z_bca_hi",
        "d_spearman", "d_spearman_bca_lo", "d_spearman_bca_hi",
    ]
    print(paired[display_cols].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nWrote analysis tables to {OUT}")

if __name__ == "__main__":
    main()
