# SPDX-License-Identifier: MIT
# Derived from https://github.com/zlisysu/Weighted_cc (Copyright (c) 2022 zlisysu).
# Modified for ProtMutMap; see protmutmap/wcc/__init__.py.
from .graphs import Graph
from . import callig as lig
import numpy as np
import pandas as pd


def wcc_from_dataframe(df, ref_mol='', ref_ene=0.0, print_pairs=False, minimum_cycles=2, mol1_col='mol1', mol2_col='mol2', ddg_col='ddG', uncertainty_col=None):
    """
    Calculate weighted cycle closure from a pandas DataFrame.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with specified columns for mol1, mol2, ddG
        Optional additional columns can contain weights/standard deviations
    ref_mol : str, optional
        Reference molecule for calculating energies. If empty, uses first molecule.
    ref_ene : float, optional
        Energy for the reference molecule. Default: 0.0
    print_pairs : bool, optional
        Whether to print pairwise energies. Default: False
    minimum_cycles : int, optional
        Minimum number of cycle closure iterations. Default: 2
    mol1_col : str, optional
        Column name for first molecule. Default: 'mol1'
    mol2_col : str, optional
        Column name for second molecule. Default: 'mol2'
    ddg_col : str, optional
        Column name for ddG values. Default: 'ddG'
    uncertainty_col : str, optional
        Column name for BAR uncertainty (std). If provided, enables weighted cycle
        closure: larger-uncertainty edges receive proportionally more correction.
        result["energies"] will have index 0 = unweighted, index 1 = weighted.
        Default: None (unweighted).

    Returns:
    --------
    dict
        Dictionary containing molecule energies, path-dependent errors, and path-independent errors
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")

    required_cols = [mol1_col, mol2_col, ddg_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"DataFrame must contain columns: {missing_cols}. Available columns: {list(df.columns)}")

    # Create graph from DataFrame
    g = Graph(dataframe=df, mol1_col=mol1_col, mol2_col=mol2_col, ddg_col=ddg_col,
              uncertainty_col=uncertainty_col)
    g.getAllCyles()

    # if len(g.cycles) == 0:
    #     print("No cycle in this graph.")
    #     return None

    # Perform cycle closure
    g.iterateCycleClosure(minimum_cycles=minimum_cycles)

    # Print pairwise energies if requested
    if print_pairs:
        g.printEnePairs()

    # Set up node mapping and reference
    node_map = lig.set_node_map(g)

    if not ref_mol.strip():
        ref_mol = g.V[0]

    try:
        ref_node = g.V.index(ref_mol)
    except ValueError:
        print(f"Check your arguments. Reference molecule '{ref_mol}' isn't in your input data!")
        return None

    # Calculate errors and energies
    path_independent_error = lig.cal_node_path_independent_error(g.V, node_map)
    path_dependent_error, path = lig.cal_node_path_dependent_error(ref_node, g.V, node_map)
    mol_ene = lig.calcMolEnes(ref_ene, g, path)

    # node_map stores edge variances (err**2; see CalLig.set_node_map), so the
    # Dijkstra distances and the per-row maxima are sums/maxes of variances. Return
    # them as kcal/mol standard errors (sqrt) so CSV consumers don't have to know
    # the internal squaring convention.
    def _to_sqrt_float(v):
        try:
            return float(v.sqrt()) if hasattr(v, 'sqrt') else float(v) ** 0.5
        except Exception:
            return float(v) ** 0.5
    path_dependent_error_sqrt = [_to_sqrt_float(v) for v in path_dependent_error]
    path_independent_error_sqrt = [_to_sqrt_float(v) for v in path_independent_error]

    # Return results as dictionary
    return {
        'molecules': g.V,
        'energies': mol_ene,
        'path_dependent_errors': path_dependent_error_sqrt,
        'path_independent_errors': path_independent_error_sqrt,
        'graph': g
    }


def _nodes_in_first_seen_order(df: pd.DataFrame, mol1_col: str, mol2_col: str) -> list[str]:
    nodes: list[str] = []
    seen: set[str] = set()
    for row in df[[mol1_col, mol2_col]].itertuples(index=False):
        for node in (str(row[0]), str(row[1])):
            if node not in seen:
                seen.add(node)
                nodes.append(node)
    return nodes


def _reference_component_edges(
    df: pd.DataFrame,
    *,
    ref_mol: str,
    mol1_col: str,
    mol2_col: str,
) -> pd.DataFrame:
    adjacency: dict[str, set[str]] = {}
    for row in df[[mol1_col, mol2_col]].itertuples(index=False):
        src = str(row[0])
        dst = str(row[1])
        adjacency.setdefault(src, set()).add(dst)
        adjacency.setdefault(dst, set()).add(src)

    if ref_mol not in adjacency:
        return df.iloc[0:0].copy()

    reachable = {ref_mol}
    stack = [ref_mol]
    while stack:
        node = stack.pop()
        for nxt in adjacency.get(node, set()):
            if nxt not in reachable:
                reachable.add(nxt)
                stack.append(nxt)

    mask = df[mol1_col].astype(str).isin(reachable) & df[mol2_col].astype(str).isin(reachable)
    return df.loc[mask].copy()


def wcc_multi_edge_from_dataframe(
    df,
    ref_mol='',
    ref_ene=0.0,
    mol1_col='mol1',
    mol2_col='mol2',
    ddg_col='ddG',
    uncertainty_col='ddG_err',
    min_sigma=0.05,
):
    """Estimate node energies from multiple edge observations.

    Unlike :func:`wcc_from_dataframe`, this row-based estimator does not collapse
    repeated ``mol1``/``mol2`` pairs into a single dictionary entry. Each valid
    row contributes one weighted least-squares observation:

        ddG_ij = E_j - E_i

    The supplied uncertainty column is interpreted as a BAR standard error in
    kcal/mol and converted to weights as ``1 / sigma**2``. Rows with missing or
    non-positive uncertainties are dropped; very small positive uncertainties are
    floored at ``min_sigma``.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")

    required_cols = [mol1_col, mol2_col, ddg_col, uncertainty_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"DataFrame must contain columns: {missing_cols}. Available columns: {list(df.columns)}")

    work = df.copy()
    endpoint_valid = work[mol1_col].notna() & work[mol2_col].notna()
    work[mol1_col] = work[mol1_col].astype(str)
    work[mol2_col] = work[mol2_col].astype(str)
    work["_wccme_ddg"] = pd.to_numeric(work[ddg_col], errors="coerce")
    work["_wccme_sigma_raw"] = pd.to_numeric(work[uncertainty_col], errors="coerce")

    valid = (
        endpoint_valid
        & np.isfinite(work["_wccme_ddg"].to_numpy(dtype=float))
        & np.isfinite(work["_wccme_sigma_raw"].to_numpy(dtype=float))
        & (work["_wccme_sigma_raw"].to_numpy(dtype=float) > 0)
    )
    dropped_edges = work.loc[~valid].copy()
    work = work.loc[valid].copy()
    if work.empty:
        return None

    if not str(ref_mol).strip():
        ref_mol = str(work.iloc[0][mol1_col])
    else:
        ref_mol = str(ref_mol)

    work = _reference_component_edges(work, ref_mol=ref_mol, mol1_col=mol1_col, mol2_col=mol2_col)
    if work.empty:
        return None

    nodes = _nodes_in_first_seen_order(work, mol1_col, mol2_col)
    if ref_mol not in nodes:
        return None
    fit_nodes = [node for node in nodes if node != ref_mol]
    col_index = {node: i for i, node in enumerate(fit_nodes)}

    design = np.zeros((len(work), len(fit_nodes)), dtype=float)
    for row_i, row in enumerate(work[[mol1_col, mol2_col]].itertuples(index=False)):
        src = str(row[0])
        dst = str(row[1])
        if src in col_index:
            design[row_i, col_index[src]] -= 1.0
        if dst in col_index:
            design[row_i, col_index[dst]] += 1.0

    values = work["_wccme_ddg"].to_numpy(dtype=float)
    sigma = np.maximum(work["_wccme_sigma_raw"].to_numpy(dtype=float), float(min_sigma))
    weights = 1.0 / (sigma ** 2)

    if design.shape[1] == 0:
        solution = np.zeros(0, dtype=float)
        fitted = np.zeros(len(work), dtype=float)
        stderr_by_node = {ref_mol: 0.0}
    else:
        sqrt_w = np.sqrt(weights)
        lhs = design * sqrt_w[:, None]
        rhs = values * sqrt_w
        solution, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)
        fitted = design @ solution
        normal = design.T @ (weights[:, None] * design)
        cov = np.linalg.pinv(normal)
        stderr = np.sqrt(np.maximum(np.diag(cov), 0.0))
        stderr_by_node = {ref_mol: 0.0}
        stderr_by_node.update({node: float(stderr[i]) for i, node in enumerate(fit_nodes)})

    rel_energy = {ref_mol: 0.0}
    rel_energy.update({node: float(solution[i]) for i, node in enumerate(fit_nodes)})
    energies = [float(ref_ene) + rel_energy[node] for node in nodes]
    energy_errs = [stderr_by_node.get(node, np.nan) for node in nodes]

    edge_out = work.drop(columns=["_wccme_ddg", "_wccme_sigma_raw"]).copy()
    edge_out["wccme_sigma"] = sigma
    edge_out["wccme_weight"] = weights
    edge_out["wccme_fitted_ddG"] = fitted
    edge_out["wccme_residual"] = values - fitted

    return {
        "molecules": nodes,
        "energies": [energies],
        "energy_errs": energy_errs,
        "path_dependent_errors": energy_errs,
        "path_independent_errors": energy_errs,
        "edges": edge_out,
        "dropped_edges": dropped_edges,
    }
