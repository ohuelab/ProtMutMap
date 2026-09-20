"""Combine forward and reverse FEP observations.

For signed forward/reverse values f and r, hysteresis is f + r and the
forward-direction average is (f - r) / 2. Assuming independent errors,
the average's standard error is sqrt(sigma_f**2 + sigma_r**2) / 2."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from protmutmap.gather_results import gather_results

def compute_bidirectional_table(
    links_df: pd.DataFrame,
    base_target_dir: Path,
    partner_chains_dict: dict,
    *,
    time_ps: float | None = None,
    allow_bar1_fallback: bool = True,
    charge_only: bool = False,
) -> pd.DataFrame:
    """Compute forward/reverse paired BAR ddG values for edges.

    Parameters
    ----------
    links_df : DataFrame with columns from_mutation, to_mutation
    base_target_dir : .../mutmap_work_large
    partner_chains_dict : {"partner1": [...], "partner2": [...]}
    time_ps : forwarded to :func:`gather_results`. When set, hysteresis and
        bidir_avg are computed from the BAR estimate at this cutoff (reads
        bar_time_series.csv per leg). Useful to compare 4 ns vs 8 ns hysteresis
        without re-running the FEP.
    charge_only : when True, only edges where ``has_charge_change == True``
        are processed.
        Default False processes all edges.

    Returns
    -------
    DataFrame with columns:
        from_mutation, to_mutation,
        ddG_f, ddG_f_err, ddG_r, ddG_r_err,
        hysteresis, bidir_avg, bidir_err
        (one row per edge with both forward and reverse FEPs available;
         edges missing reverse FEP directories are silently skipped)."""
    if links_df.empty:
        return pd.DataFrame()

    if charge_only:
        if "has_charge_change" not in links_df.columns:
            return pd.DataFrame()
        all_df = links_df[links_df["has_charge_change"] == True].copy()  # noqa: E712
        if all_df.empty:
            return pd.DataFrame()
    else:
        all_df = links_df.copy()

    forward_df = gather_results(
        all_df, base_target_dir, partner_chains_dict,
        time_ps=time_ps, allow_bar1_fallback=allow_bar1_fallback,
    )
    if forward_df is None or forward_df.empty:
        return pd.DataFrame()

    # Build reverse edges by swapping from/to. Drop derived columns so
    # gather_results recomputes them from the swapped from/to.
    reverse_input = all_df.copy()
    reverse_input["from_mutation"], reverse_input["to_mutation"] = (
        all_df["to_mutation"].values,
        all_df["from_mutation"].values,
    )
    reverse_input = reverse_input.drop(columns=["mutation_diff"], errors="ignore")
    reverse_df = gather_results(
        reverse_input, base_target_dir, partner_chains_dict,
        time_ps=time_ps, allow_bar1_fallback=allow_bar1_fallback,
    )

    rows: list[dict] = []
    for _, fwd in forward_df.iterrows():
        ddG_f = fwd.get("calc_ddG", np.nan)
        sig_f = fwd.get("calc_ddG_err", np.nan)
        if pd.isna(ddG_f):
            continue

        rev = reverse_df[
            (reverse_df["from_mutation"] == fwd["to_mutation"])
            & (reverse_df["to_mutation"] == fwd["from_mutation"])
        ]
        if rev.empty:
            continue
        rev_row = rev.iloc[0]
        ddG_r = rev_row.get("calc_ddG", np.nan)
        sig_r = rev_row.get("calc_ddG_err", np.nan)
        if pd.isna(ddG_r):
            continue

        hysteresis = float(ddG_f) + float(ddG_r)
        bidir_avg = (float(ddG_f) - float(ddG_r)) / 2.0
        if pd.isna(sig_f) or pd.isna(sig_r):
            bidir_err = np.nan
        else:
            bidir_err = float(np.sqrt(float(sig_f) ** 2 + float(sig_r) ** 2) / 2.0)

        rows.append({
            "from_mutation": fwd["from_mutation"],
            "to_mutation": fwd["to_mutation"],
            "ddG_f": float(ddG_f),
            "ddG_f_err": float(sig_f) if not pd.isna(sig_f) else np.nan,
            "ddG_r": float(ddG_r),
            "ddG_r_err": float(sig_r) if not pd.isna(sig_r) else np.nan,
            "hysteresis": hysteresis,
            "bidir_avg": bidir_avg,
            "bidir_err": bidir_err,
        })
    return pd.DataFrame(rows)

def build_bidir_lookup(bidir_df: pd.DataFrame) -> dict:
    """{(from, to): (bidir_avg, bidir_err)} dict for fast Additive_bidirectional lookups."""
    if bidir_df is None or bidir_df.empty:
        return {}
    return {
        (r["from_mutation"], r["to_mutation"]): (
            float(r["bidir_avg"]),
            float(r["bidir_err"]) if not pd.isna(r["bidir_err"]) else np.nan,
        )
        for _, r in bidir_df.iterrows()
    }

def apply_bidirectional_to_calc_df(
    calc_df: pd.DataFrame | None,
    bidir_df: pd.DataFrame,
) -> pd.DataFrame | None:
    """Return a copy of calc_df with charge-edge calc_ddG/calc_ddG_err replaced
    by bidir_avg/bidir_err. Non-charge edges are untouched.

    If bidir_df is empty or calc_df is None/empty, returns calc_df unchanged.
    """
    if calc_df is None or calc_df.empty or bidir_df is None or bidir_df.empty:
        return calc_df

    out = calc_df.copy()
    lookup = build_bidir_lookup(bidir_df)
    for idx, row in out.iterrows():
        key = (row["from_mutation"], row["to_mutation"])
        if key in lookup:
            avg, err = lookup[key]
            out.at[idx, "calc_ddG"] = avg
            if "calc_ddG_err" in out.columns:
                out.at[idx, "calc_ddG_err"] = err
    return out

def expand_bidirectional_to_observations(
    calc_df: pd.DataFrame | None,
    bidir_df: pd.DataFrame,
) -> pd.DataFrame | None:
    """Return edge observations for WCC multi-edge estimation.

    Existing bidirectional handling averages a paired forward/reverse charge edge
    into one corrected edge. WCCME instead treats the two BAR estimates as two
    separate observations:

    - ``from_mutation -> to_mutation`` with ``ddG_f`` / ``ddG_f_err``
    - ``to_mutation -> from_mutation`` with ``ddG_r`` / ``ddG_r_err``

    Non-paired edges are passed through unchanged. If optional grouping columns
    such as ``source`` or ``system`` are present in both inputs, they are included
    in the pairing key so results from different systems are never mixed.
    """
    if calc_df is None or calc_df.empty:
        return calc_df

    out = calc_df.copy()
    out["wccme_bidir_replaced"] = False
    out["wccme_observation"] = "single"
    out["wccme_pair_id"] = ""
    out["wccme_hysteresis"] = np.nan

    if bidir_df is None or bidir_df.empty:
        return out

    endpoint_cols = ["from_mutation", "to_mutation"]
    if any(col not in out.columns for col in endpoint_cols):
        raise ValueError("calc_df must contain from_mutation and to_mutation")
    if any(col not in bidir_df.columns for col in endpoint_cols):
        raise ValueError("bidir_df must contain from_mutation and to_mutation")

    required_bidir = ["ddG_f", "ddG_f_err", "ddG_r", "ddG_r_err", "hysteresis"]
    missing = [col for col in required_bidir if col not in bidir_df.columns]
    if missing:
        raise ValueError(f"bidir_df missing required columns: {missing}")

    group_cols = [col for col in ("source", "system") if col in out.columns and col in bidir_df.columns]
    key_cols = group_cols + endpoint_cols
    base = out.reset_index(drop=True)
    indexed = {tuple(row[col] for col in key_cols): i for i, row in base.iterrows()}

    forward_keys = set()
    reverse_keys = set()
    replacement_rows: list[pd.Series] = []

    for _, bidir in bidir_df.iterrows():
        src = bidir["from_mutation"]
        dst = bidir["to_mutation"]
        prefix = tuple(bidir[col] for col in group_cols)
        fwd_key = prefix + (src, dst)
        rev_key = prefix + (dst, src)
        if fwd_key not in indexed:
            continue

        forward_keys.add(fwd_key)
        reverse_keys.add(rev_key)
        template = base.iloc[indexed[fwd_key]].copy()
        prefix_label = "|".join(str(v) for v in prefix)
        pair_id = f"{prefix_label}|{src}->{dst}" if prefix_label else f"{src}->{dst}"

        fwd = template.copy()
        fwd["from_mutation"] = src
        fwd["to_mutation"] = dst
        fwd["calc_ddG"] = float(bidir["ddG_f"])
        fwd["calc_ddG_err"] = float(bidir["ddG_f_err"])
        fwd["wccme_bidir_replaced"] = True
        fwd["wccme_observation"] = "forward"
        fwd["wccme_pair_id"] = pair_id
        fwd["wccme_hysteresis"] = float(bidir["hysteresis"])

        rev = template.copy()
        rev["from_mutation"] = dst
        rev["to_mutation"] = src
        rev["calc_ddG"] = float(bidir["ddG_r"])
        rev["calc_ddG_err"] = float(bidir["ddG_r_err"])
        rev["wccme_bidir_replaced"] = True
        rev["wccme_observation"] = "reverse"
        rev["wccme_pair_id"] = pair_id
        rev["wccme_hysteresis"] = float(bidir["hysteresis"])

        replacement_rows.extend([fwd, rev])

    if not replacement_rows:
        return out

    replaced_keys = forward_keys | reverse_keys

    def is_replaced(row: pd.Series) -> bool:
        return tuple(row[col] for col in key_cols) in replaced_keys

    keep = out.loc[~out.apply(is_replaced, axis=1)].copy()
    repl = pd.DataFrame(replacement_rows, columns=out.columns)
    return pd.concat([keep, repl], ignore_index=True)
