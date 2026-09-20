"""Robust graph estimators for mutation-network FEP edges.

These utilities estimate node potentials directly from edge ddG observations.
Input: edge ΔΔG values. Output: node potentials and edge diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Iterable

import networkx as nx
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class RobustFitResult:
    """Container for node-potential and edge-residual outputs."""

    nodes: pd.DataFrame
    edges: pd.DataFrame
    converged: bool
    n_iter: int

def _finite_positive(values: Iterable[float]) -> list[float]:
    out = []
    for value in values:
        try:
            v = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(v) and v > 0:
            out.append(v)
    return out

def _sanitize_sigma(
    df: pd.DataFrame,
    err_col: str | None,
    *,
    default_sigma: float | None,
    min_sigma: float,
) -> np.ndarray:
    if err_col is not None and err_col in df.columns:
        raw = pd.to_numeric(df[err_col], errors="coerce").to_numpy(dtype=float)
        finite = _finite_positive(raw)
    else:
        raw = np.full(len(df), np.nan, dtype=float)
        finite = []

    fallback = default_sigma
    if fallback is None:
        fallback = float(np.median(finite)) if finite else 1.0
    fallback = max(float(fallback), min_sigma)

    sigma = np.where(np.isfinite(raw) & (raw > 0), raw, fallback)
    return np.maximum(sigma.astype(float), float(min_sigma))

def _reference_component(
    df: pd.DataFrame,
    *,
    from_col: str,
    to_col: str,
    ref_node: str,
) -> pd.DataFrame:
    graph = nx.Graph()
    graph.add_edges_from(zip(df[from_col].astype(str), df[to_col].astype(str)))
    if not graph.nodes:
        return df.iloc[0:0].copy()
    if ref_node not in graph:
        return df.iloc[0:0].copy()

    keep_nodes = nx.node_connected_component(graph, ref_node)
    mask = df[from_col].astype(str).isin(keep_nodes) & df[to_col].astype(str).isin(keep_nodes)
    return df.loc[mask].copy()

def _build_design(
    df: pd.DataFrame,
    *,
    from_col: str,
    to_col: str,
    value_col: str,
    ref_node: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    nodes = sorted(set(df[from_col].astype(str)) | set(df[to_col].astype(str)))
    fit_nodes = [node for node in nodes if node != ref_node]
    col_index = {node: i for i, node in enumerate(fit_nodes)}

    design = np.zeros((len(df), len(fit_nodes)), dtype=float)
    for row_i, row in enumerate(df.itertuples(index=False)):
        src = str(getattr(row, from_col))
        dst = str(getattr(row, to_col))
        if src in col_index:
            design[row_i, col_index[src]] -= 1.0
        if dst in col_index:
            design[row_i, col_index[dst]] += 1.0
    values = pd.to_numeric(df[value_col], errors="coerce").to_numpy(dtype=float)
    return design, values, fit_nodes

def _solve_weighted_lstsq(
    design: np.ndarray,
    values: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    if design.shape[1] == 0:
        return np.zeros(0, dtype=float)
    sqrt_w = np.sqrt(np.maximum(weights, 0.0))
    lhs = design * sqrt_w[:, None]
    rhs = values * sqrt_w
    solution, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)
    return solution

def _node_stderr(
    design: np.ndarray,
    weights: np.ndarray,
    fit_nodes: list[str],
    *,
    ref_node: str,
) -> dict[str, float]:
    out = {ref_node: 0.0}
    if design.shape[1] == 0:
        return out
    normal = design.T @ (weights[:, None] * design)
    cov = np.linalg.pinv(normal)
    stderr = np.sqrt(np.maximum(np.diag(cov), 0.0))
    out.update({node: float(stderr[i]) for i, node in enumerate(fit_nodes)})
    return out

IRLS_METHODS = {"huber", "l1", "soft_l1", "cauchy", "tukey"}

def _irls_robust_factor(method: str, z: np.ndarray, delta: float, l1_eps: float = 1e-3) -> np.ndarray:
    """Return the IRLS weight multiplier w(z) for standardized residuals z.

    ``z = |residual| / sigma``. For "huber" this reproduces the pre-existing
    behavior exactly (weight 1 below delta, delta/z above; internal epsilon is
    a fixed 1e-12, unrelated to ``l1_eps``). The other kinds are additional
    M-estimator loss families sharing the same IRLS loop. ``l1_eps`` is the
    floor on z used only by ``"l1"``; the pure 1/z weight oscillates near-zero
    residuals, so a larger floor (default 1e-3) damps that instability.
    """
    eps = 1e-12
    if method == "huber":
        robust_factor = np.ones_like(z)
        mask = z > delta
        robust_factor[mask] = delta / np.maximum(z[mask], eps)
        return robust_factor
    if method == "l1":
        return 1.0 / np.maximum(z, l1_eps)
    if method == "soft_l1":
        return 1.0 / np.sqrt(1.0 + (z / delta) ** 2)
    if method == "cauchy":
        return 1.0 / (1.0 + (z / delta) ** 2)
    if method == "tukey":
        robust_factor = np.zeros_like(z)
        mask = z <= delta
        robust_factor[mask] = (1.0 - (z[mask] / delta) ** 2) ** 2
        return robust_factor
    raise ValueError(f"unknown IRLS method: {method!r}")

def fit_node_potentials(
    edges_df: pd.DataFrame,
    *,
    from_col: str = "from_mutation",
    to_col: str = "to_mutation",
    value_col: str = "calc_ddG",
    err_col: str | None = "calc_ddG_err",
    ref_node: str = "WT",
    method: str = "huber",
    huber_delta: float = 1.5,
    max_iter: int = 50,
    tol: float = 1e-7,
    min_sigma: float = 0.05,
    default_sigma: float | None = None,
    edge_weight_col: str | None = None,
    l1_eps: float = 1e-3,
) -> RobustFitResult:
    """Estimate mutation-node potentials from FEP edge ddGs.

    The model is y_ij = E_j - E_i + residual_ij with E_ref_node fixed to zero.
    ``method="wls"`` uses weighted least squares. ``method="huber"`` uses IRLS
    with Huber weights on standardized residuals. ``"l1"``, ``"soft_l1"``,
    ``"cauchy"``, and ``"tukey"`` are additional M-estimator loss families that
    share the same IRLS loop (``huber_delta`` is reused as their scale
    parameter, except ``"l1"`` which has no scale). ``l1_eps`` is a floor on
    the standardized residual used only by ``method="l1"``, to damp the
    1/z-weight oscillation near zero residuals; it does not affect any other
    method. ``edge_weight_col`` can supply an additional confidence multiplier
    such as cycle-derived downweights.
    """
    required = {from_col, to_col, value_col}
    missing = sorted(required - set(edges_df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = edges_df.dropna(subset=[from_col, to_col, value_col]).copy()
    if df.empty:
        return RobustFitResult(pd.DataFrame(), pd.DataFrame(), True, 0)
    df[from_col] = df[from_col].astype(str)
    df[to_col] = df[to_col].astype(str)
    df = _reference_component(df, from_col=from_col, to_col=to_col, ref_node=ref_node)
    if df.empty:
        return RobustFitResult(pd.DataFrame(), pd.DataFrame(), False, 0)

    design, values, fit_nodes = _build_design(
        df,
        from_col=from_col,
        to_col=to_col,
        value_col=value_col,
        ref_node=ref_node,
    )
    sigma = _sanitize_sigma(df, err_col, default_sigma=default_sigma, min_sigma=min_sigma)

    if edge_weight_col and edge_weight_col in df.columns:
        extra_weight = pd.to_numeric(df[edge_weight_col], errors="coerce").to_numpy(dtype=float)
        extra_weight = np.where(np.isfinite(extra_weight) & (extra_weight > 0), extra_weight, 0.0)
    else:
        extra_weight = np.ones(len(df), dtype=float)
    base_weights = extra_weight / (sigma ** 2)

    if method not in {"wls"} | IRLS_METHODS:
        raise ValueError(f"method must be one of {{'wls'}} | {sorted(IRLS_METHODS)}")

    solution = _solve_weighted_lstsq(design, values, base_weights)
    final_weights = base_weights.copy()
    converged = method == "wls"
    n_iter = 1

    if method in IRLS_METHODS:
        previous = solution.copy()
        for n_iter in range(1, max_iter + 1):
            residual = values - design @ solution
            z = np.abs(residual) / sigma
            robust_factor = _irls_robust_factor(method, z, huber_delta, l1_eps=l1_eps)
            final_weights = base_weights * robust_factor
            solution = _solve_weighted_lstsq(design, values, final_weights)
            step = np.max(np.abs(solution - previous)) if solution.size else 0.0
            if step < tol:
                converged = True
                break
            previous = solution.copy()

    energy = {ref_node: 0.0}
    energy.update({node: float(solution[i]) for i, node in enumerate(fit_nodes)})
    stderr = _node_stderr(design, final_weights, fit_nodes, ref_node=ref_node)

    fitted = design @ solution
    residual = values - fitted
    nodes = pd.DataFrame(
        {
            "node": list(energy.keys()),
            "energy": list(energy.values()),
            "energy_err": [stderr.get(node, np.nan) for node in energy],
        }
    )
    edge_out = df.copy()
    edge_out["node_fit_ddG"] = fitted
    edge_out["node_fit_residual"] = residual
    edge_out["node_fit_sigma"] = sigma
    edge_out["node_fit_weight"] = final_weights
    edge_out["node_fit_base_weight"] = base_weights
    edge_out["node_fit_robust_factor"] = np.divide(
        final_weights,
        base_weights,
        out=np.zeros_like(final_weights),
        where=base_weights > 0,
    )

    return RobustFitResult(nodes=nodes, edges=edge_out, converged=converged, n_iter=n_iter)

def _signed_lookup(
    df: pd.DataFrame,
    *,
    from_col: str,
    to_col: str,
    value_col: str,
    err_col: str | None,
    min_sigma: float,
    default_sigma: float | None = None,
) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], float]]:
    sigma = _sanitize_sigma(df, err_col, default_sigma=default_sigma, min_sigma=min_sigma)
    values: dict[tuple[str, str], float] = {}
    sigmas: dict[tuple[str, str], float] = {}
    for i, row in enumerate(df.itertuples(index=False)):
        src = str(getattr(row, from_col))
        dst = str(getattr(row, to_col))
        value = float(getattr(row, value_col))
        values[(src, dst)] = value
        sigmas[(src, dst)] = float(sigma[i])
        if (dst, src) not in values:
            values[(dst, src)] = -value
            sigmas[(dst, src)] = float(sigma[i])
    return values, sigmas

def _path_sum(
    path: list[str],
    values: dict[tuple[str, str], float],
    sigmas: dict[tuple[str, str], float],
) -> tuple[float, float] | None:
    total = 0.0
    var = 0.0
    for src, dst in zip(path[:-1], path[1:]):
        key = (src, dst)
        if key not in values:
            return None
        total += values[key]
        var += sigmas[key] ** 2
    return total, sqrt(var)

def cycle_residual_edge_scores(
    edges_df: pd.DataFrame,
    *,
    from_col: str = "from_mutation",
    to_col: str = "to_mutation",
    value_col: str = "calc_ddG",
    err_col: str | None = "calc_ddG_err",
    max_path_len: int = 6,
    max_paths_per_edge: int = 64,
    min_sigma: float = 0.05,
    default_sigma: float | None = None,
    min_weight_factor: float = 0.05,
    weight_alpha: float = 1.0,
) -> pd.DataFrame:
    """Score each edge against alternative paths that exclude that edge.

    For edge i->j, enumerate simple i->j paths after removing the direct
    undirected edge. The median normalized direct-vs-path mismatch is returned
    as ``cycle_q``. Edges with no alternative path get ``cycle_q=0`` and are not
    downweighted because the graph provides no independent check.
    """
    required = {from_col, to_col, value_col}
    missing = sorted(required - set(edges_df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = edges_df.dropna(subset=[from_col, to_col, value_col]).copy()
    if df.empty:
        return pd.DataFrame()
    df[from_col] = df[from_col].astype(str)
    df[to_col] = df[to_col].astype(str)

    graph = nx.Graph()
    graph.add_edges_from(zip(df[from_col], df[to_col]))
    values, sigmas = _signed_lookup(
        df,
        from_col=from_col,
        to_col=to_col,
        value_col=value_col,
        err_col=err_col,
        min_sigma=min_sigma,
        default_sigma=default_sigma,
    )

    rows = []
    for row in df.itertuples(index=False):
        src = str(getattr(row, from_col))
        dst = str(getattr(row, to_col))
        direct = values[(src, dst)]
        direct_sigma = sigmas[(src, dst)]

        g2 = graph.copy()
        if g2.has_edge(src, dst):
            g2.remove_edge(src, dst)

        diffs = []
        norm_diffs = []
        if src in g2 and dst in g2:
            try:
                paths = nx.all_simple_paths(g2, src, dst, cutoff=max_path_len)
                for n_path, path in enumerate(paths):
                    if n_path >= max_paths_per_edge:
                        break
                    alt = _path_sum(path, values, sigmas)
                    if alt is None:
                        continue
                    alt_value, alt_sigma = alt
                    diff = direct - alt_value
                    denom = max(sqrt(direct_sigma ** 2 + alt_sigma ** 2), min_sigma)
                    diffs.append(abs(diff))
                    norm_diffs.append(abs(diff) / denom)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                pass

        n_alt = len(norm_diffs)
        median_abs = float(np.median(diffs)) if diffs else 0.0
        q = float(np.median(norm_diffs)) if norm_diffs else 0.0
        weight_factor = max(float(min_weight_factor), 1.0 / (1.0 + float(weight_alpha) * q ** 2))
        rows.append(
            {
                from_col: src,
                to_col: dst,
                "cycle_alt_n_paths": n_alt,
                "cycle_alt_median_abs_residual": median_abs,
                "cycle_q": q,
                "cycle_weight_factor": weight_factor,
            }
        )

    return pd.DataFrame(rows)
