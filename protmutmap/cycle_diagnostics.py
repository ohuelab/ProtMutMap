"""Compute cycle hysteresis and edge inconsistency scores.

An edge score is the median absolute cycle hysteresis divided by cycle length,
over cycles containing that edge. Edges belonging to the same cycles can
have identical scores; cycle closure alone cannot identify a faulty edge."""

from __future__ import annotations

import statistics
from typing import Iterable

import pandas as pd

from protmutmap.wcc.graphs import Graph

def _cycle_edges(cycle_nodes: list[str]) -> list[tuple[str, str]]:
    """Return list of consecutive node pairs in a cycle path (last connects to first)."""
    edges = []
    for i in range(len(cycle_nodes) - 1):
        edges.append((cycle_nodes[i], cycle_nodes[i + 1]))
    return edges

def _signed_ddg(edge_ddg: dict, mol1: str, mol2: str) -> float | None:
    """Return ddG(mol1->mol2) using a directed-edge ddG dict; flip sign if reversed."""
    if (mol1, mol2) in edge_ddg:
        return edge_ddg[(mol1, mol2)]
    if (mol2, mol1) in edge_ddg:
        v = edge_ddg[(mol2, mol1)]
        return -v if v is not None else None
    return None

def compute_edge_cycle_scores(
    edges_df: pd.DataFrame,
    *,
    from_col: str = "from_mutation",
    to_col: str = "to_mutation",
    ddg_col: str = "calc_ddG",
    err_col: str | None = "calc_ddG_err",
    extra_cols: Iterable[str] = ("has_charge_change",),
    max_cycle_size: int | None = 6,
) -> pd.DataFrame:
    """Return per-edge cycle diagnostics.

    Output columns:
        from_mutation, to_mutation, calc_ddG, calc_ddG_err, has_charge_change (if present),
        n_cycles            -- number of cycles this edge participates in
        edge_cycle_score    -- median of |delta|/N_edges over those cycles (kcal/mol)
        max_cycle_delta     -- max |delta| of any cycle this edge sits on
        max_cycle_size      -- size (#edges) of that worst cycle
        cycle_score_tied_n  -- number of other directed edges with the same edge_cycle_score
                                (identifiability hint)
    """
    df = edges_df.dropna(subset=[ddg_col]).copy()
    if df.empty:
        return pd.DataFrame()

    # Build undirected graph for cycle enumeration. We use the ddg_col only; the Graph
    # stores both (a,b) and (b,a) so cycle traversal sign-handles automatically via
    # Graph.ddG_cc (which mirrors ddg_col with appropriate sign).
    graph = Graph(
        dataframe=df[[from_col, to_col, ddg_col]],
        mol1_col=from_col,
        mol2_col=to_col,
        ddg_col=ddg_col,
    )
    graph.getAllCyles()

    # ddG lookup keyed on directed pair, sourced directly from df (not Graph.ddG_cc
    # which gets mutated by cycle closure if called later).
    edge_ddg: dict[tuple[str, str], float] = {}
    for _, row in df.iterrows():
        a, b = str(row[from_col]), str(row[to_col])
        edge_ddg[(a, b)] = float(row[ddg_col])

    # For each cycle, compute |delta| and size.
    per_edge_contribs: dict[tuple[str, str], list[float]] = {}
    per_edge_max: dict[tuple[str, str], tuple[float, int]] = {}
    for cycle_nodes in graph.cycles:
        # mutmap stores cycles as paths a -> b -> c -> ... -> a (last node = first).
        edges = _cycle_edges(cycle_nodes)
        n_edges = len(edges)
        if max_cycle_size is not None and n_edges > max_cycle_size:
            continue
        delta = 0.0
        ok = True
        for (m1, m2) in edges:
            v = _signed_ddg(edge_ddg, m1, m2)
            if v is None:
                ok = False
                break
            delta += v
        if not ok or n_edges == 0:
            continue
        contrib = abs(delta) / n_edges
        for (m1, m2) in edges:
            # Normalize to the directed key present in df (so output matches input)
            key = (m1, m2) if (m1, m2) in edge_ddg else (m2, m1)
            per_edge_contribs.setdefault(key, []).append(contrib)
            cur_max = per_edge_max.get(key)
            if cur_max is None or abs(delta) > cur_max[0]:
                per_edge_max[key] = (abs(delta), n_edges)

    # Assemble output rows in the input df order.
    cols_in = [from_col, to_col, ddg_col]
    if err_col and err_col in df.columns:
        cols_in.append(err_col)
    for c in extra_cols:
        if c in df.columns and c not in cols_in:
            cols_in.append(c)

    rows: list[dict] = []
    for _, row in df.iterrows():
        key = (str(row[from_col]), str(row[to_col]))
        contribs = per_edge_contribs.get(key, [])
        n_cyc = len(contribs)
        score = statistics.median(contribs) if contribs else 0.0
        max_d, max_size = per_edge_max.get(key, (0.0, 0))
        out = {c: row[c] for c in cols_in}
        out["n_cycles"] = n_cyc
        out["edge_cycle_score"] = float(score)
        out["max_cycle_delta"] = float(max_d)
        out["max_cycle_size"] = int(max_size)
        rows.append(out)

    out_df = pd.DataFrame(rows)
    if not out_df.empty and "edge_cycle_score" in out_df.columns:
        # Tie hint: how many other edges share this score (within rounding)?
        # Helps consumers understand identifiability limits before acting on the score.
        rounded = out_df["edge_cycle_score"].round(6)
        tied_counts = rounded.map(rounded.value_counts())
        out_df["cycle_score_tied_n"] = (tied_counts - 1).astype(int)

    return out_df

def write_edge_diagnostics_csv(
    diagnostics_by_system: dict[str, pd.DataFrame],
    output_path,
) -> None:
    """Concatenate per-system diagnostics with a 'system' column and save to CSV."""
    pieces = []
    for system, df in diagnostics_by_system.items():
        if df is None or df.empty:
            continue
        d = df.copy()
        d.insert(0, "system", system)
        pieces.append(d)
    if not pieces:
        pd.DataFrame().to_csv(output_path, index=False)
        return
    combined = pd.concat(pieces, ignore_index=True)
    combined.to_csv(output_path, index=False)
