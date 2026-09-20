#!/usr/bin/env python3
"""Compute cycle-closure diagnostics and edge-error comparisons from FEP tables."""

import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

MUTMAP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MUTMAP_ROOT))

from protmutmap.wcc.graphs import Graph
from protmutmap.tools import MutationList
from protmutmap.tools.lambda_calculator import assess_mutation_difficulty

SYSTEM_CATEGORY = {
    "1A22": "Pathway", "1AO7": "Pathway", "1BJ1": "Pathway",
    "2B2X": "Pathway", "2J0T": "Pathway",
    "1CHO": "SingleRef", "1R0R": "SingleRef", "3SGB": "SingleRef", "4OFY": "SingleRef",
    "1AHW": "TargetOnly", "3SE3": "TargetOnly", "3EG5": "TargetOnly", "1MHP": "TargetOnly",
}

def is_step_difficult(from_node: str, to_node: str) -> bool:
    """Return True if the mutation step involves a difficult residue (F/Y/W/P)."""
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

def count_step_mutations(from_node: str, to_node: str) -> int:
    """Return number of mutations changed in this edge step."""
    from_set = set() if from_node == "WT" else set(from_node.split(","))
    to_set = set() if to_node == "WT" else set(to_node.split(","))
    return len((from_set - to_set) | (to_set - from_set))

def compute_cycle_closure_errors(links_df: pd.DataFrame) -> dict:
    """
    Run one cycle-closure iteration to extract per-edge cycle closure errors.

    Returns dict: {(from_node, to_node): max_cycle_closure_error}
    Edges not participating in any cycle have error = -1 (unchanged from init).
    """
    valid_df = links_df.dropna(subset=["calc_ddG"])
    if valid_df.empty:
        return {}

    g = Graph(
        dataframe=valid_df,
        mol1_col="from_mutation",
        mol2_col="to_mutation",
        ddg_col="calc_ddG",
    )
    g.getAllCyles()
    g.iterateCycleClosure(minimum_cycles=2)

    cycle_errs = {}
    for (m1, m2), err_val in g.err.items():
        cycle_errs[(m1, m2)] = float(err_val)

    return cycle_errs

def get_expected_ddg(from_node: str, to_node: str, dgs: dict) -> float:
    """
    Compute expected edge ddG from experimental dGs.json.
    Returns NaN if either node is missing from dGs.
    """
    from_dg = dgs.get(from_node, np.nan)
    to_dg = dgs.get(to_node, np.nan)
    if np.isnan(from_dg) or np.isnan(to_dg):
        return np.nan
    if from_node == "WT":
        from_dg = 0.0
    if to_node == "WT":
        to_dg = 0.0
    return to_dg - from_dg

def analyze_system(system_name: str, sys_dir: Path) -> list[dict]:
    """Analyze all edges in one system. Returns list of row dicts."""
    network_dir = sys_dir / "network"
    links_ddg_tsv = network_dir / "links_with_ddg.tsv"
    dgs_json = sys_dir / "dGs.json"

    if not links_ddg_tsv.exists():
        print(f"  {system_name}: links_with_ddg.tsv not found, skipping")
        return []

    links_df = pd.read_csv(links_ddg_tsv, sep="\t")

    dgs = {}
    if dgs_json.exists():
        with open(dgs_json) as f:
            dgs_raw = json.load(f)
        dgs = {k: float(v) for k, v in dgs_raw.items() if k != "WT"}
        dgs["WT"] = 0.0

    # Compute cycle closure errors
    cycle_errs = compute_cycle_closure_errors(links_df)

    rows = []
    for _, row in links_df.iterrows():
        from_node = str(row["from_mutation"])
        to_node = str(row["to_mutation"])
        calc_ddg = float(row["calc_ddG"]) if not pd.isna(row.get("calc_ddG", np.nan)) else np.nan
        calc_ddg_err = (
            float(row["calc_ddG_err"])
            if "calc_ddG_err" in row and not pd.isna(row["calc_ddG_err"])
            else np.nan
        )

        # Expected ddG from experimental dGs.json
        expected_ddg = get_expected_ddg(from_node, to_node, dgs)
        exp_vs_calc = abs(calc_ddg - expected_ddg) if not np.isnan(expected_ddg) and not np.isnan(calc_ddg) else np.nan

        # Cycle closure error for this edge
        fwd_key = (from_node, to_node)
        rev_key = (to_node, from_node)
        cc_err = cycle_errs.get(fwd_key, cycle_errs.get(rev_key, -1.0))
        in_cycle = cc_err > 0

        # Edge classification
        difficult = is_step_difficult(from_node, to_node)
        n_step_muts = count_step_mutations(from_node, to_node)
        n_from_muts = 0 if from_node == "WT" else len(from_node.split(","))
        n_to_muts = 0 if to_node == "WT" else len(to_node.split(","))

        rows.append({
            "system": system_name,
            "category": SYSTEM_CATEGORY.get(system_name, "Unknown"),
            "from_mutation": from_node,
            "to_mutation": to_node,
            "n_from_muts": n_from_muts,
            "n_to_muts": n_to_muts,
            "n_step_muts": n_step_muts,
            "calc_ddG": calc_ddg,
            "calc_ddG_err": calc_ddg_err,
            "expected_ddg": expected_ddg,
            "exp_vs_calc_err": exp_vs_calc,
            "cycle_closure_err": cc_err if in_cycle else np.nan,
            "in_cycle": in_cycle,
            "is_difficult": difficult,
        })

    return rows

def print_validated_edges(df: pd.DataFrame) -> None:
    """Print table of edges where experimental reference is available."""
    from scipy import stats as scipy_stats

    known = df.dropna(subset=["exp_vs_calc_err", "expected_ddg", "calc_ddG"]).copy()
    if known.empty:
        print("\n=== Validated Edges: no experimental reference data available ===")
        return

    print("\n=== Validated Edges (FEP calc vs experimental ddG) ===")
    header = (f"{'System':<6}  {'from':30s} -> {'to':30s}  "
              f"{'calc':>6}  {'exp':>6}  {'|err|':>6}  {'BAR_err':>7}")
    sep = "-" * len(header)

    for system in sorted(known["system"].unique()):
        sub = known[known["system"] == system].sort_values("exp_vs_calc_err", ascending=False)
        n = len(sub)
        rmse = np.sqrt((sub["exp_vs_calc_err"] ** 2).mean())
        mae = sub["exp_vs_calc_err"].mean()
        if n >= 2:
            r, _ = scipy_stats.pearsonr(sub["expected_ddg"], sub["calc_ddG"])
            r_str = f"{r:.3f}"
        else:
            r_str = "  N/A"
        print(f"\n  [{system}]  N={n}  RMSE={rmse:.2f}  MAE={mae:.2f}  R={r_str}")
        print("  " + sep)
        print("  " + header)
        print("  " + sep)
        for _, r in sub.iterrows():
            bar_str = f"{r['calc_ddG_err']:.2f}" if not np.isnan(r['calc_ddG_err']) else "   N/A"
            print(
                f"  {r['system']:<6}  {str(r['from_mutation']):30s} -> {str(r['to_mutation']):30s}"
                f"  {r['calc_ddG']:+6.2f}  {r['expected_ddg']:+6.2f}"
                f"  {r['exp_vs_calc_err']:6.2f}  {bar_str:>7}"
            )
        print("  " + sep)

    # Overall stats
    n_total = len(known)
    rmse_all = np.sqrt((known["exp_vs_calc_err"] ** 2).mean())
    mae_all = known["exp_vs_calc_err"].mean()
    if n_total >= 2:
        r_all, _ = scipy_stats.pearsonr(known["expected_ddg"], known["calc_ddG"])
        r_all_str = f"{r_all:.3f}"
    else:
        r_all_str = "  N/A"
    print(f"\n  [TOTAL]  N={n_total}  RMSE={rmse_all:.2f}  MAE={mae_all:.2f}  R={r_all_str}")

def print_error_by_from_nmuts(df: pd.DataFrame) -> None:
    """Print error statistics grouped by number of mutations in the from_node."""
    known = df.dropna(subset=["exp_vs_calc_err"]).copy()
    if known.empty:
        print("\n=== Error by from_node mutation count: no experimental data ===")
        return

    print("\n=== Error by from_node mutation count (n_from_muts) ===")
    print(f"\n  {'n_from_muts':>11}  {'N':>4}  {'mean|err|':>9}  {'median|err|':>11}  "
          f"{'max|err|':>8}  {'BAR_err mean':>12}")
    print("  " + "-" * 65)

    max_n = int(known["n_from_muts"].max())
    for n in range(0, max_n + 1):
        grp = known[known["n_from_muts"] == n]
        if grp.empty:
            continue
        mean_err = grp["exp_vs_calc_err"].mean()
        med_err = grp["exp_vs_calc_err"].median()
        max_err = grp["exp_vs_calc_err"].max()
        bar_mean = grp["calc_ddG_err"].mean()
        bar_str = f"{bar_mean:.2f}" if not np.isnan(bar_mean) else "   N/A"
        label = f"{n}" if n < 3 else f"≥{n}"
        print(
            f"  {label:>11}  {len(grp):>4}  {mean_err:>9.2f}  {med_err:>11.2f}"
            f"  {max_err:>8.2f}  {bar_str:>12}"
        )

    # Group ≥3 together if max_n >= 3
    if max_n >= 3:
        grp3 = known[known["n_from_muts"] >= 3]
        if not grp3.empty and not (known["n_from_muts"] == max_n).all():
            mean_err = grp3["exp_vs_calc_err"].mean()
            med_err = grp3["exp_vs_calc_err"].median()
            max_err = grp3["exp_vs_calc_err"].max()
            bar_mean = grp3["calc_ddG_err"].mean()
            bar_str = f"{bar_mean:.2f}" if not np.isnan(bar_mean) else "   N/A"
            print(
                f"  {'≥3 (all)':>11}  {len(grp3):>4}  {mean_err:>9.2f}  {med_err:>11.2f}"
                f"  {max_err:>8.2f}  {bar_str:>12}"
            )

    print("\n  Hypothesis: n_from_muts=0 (WT start) should have lower error than n≥1.")

def print_error_by_difficulty(df: pd.DataFrame) -> None:
    """Print error statistics grouped by whether the edge involves a difficult residue (F/Y/W/P)."""
    known = df.dropna(subset=["exp_vs_calc_err"]).copy()
    if known.empty:
        print("\n=== Error by mutation difficulty: no experimental data ===")
        return

    print("\n=== Error by mutation difficulty (F/Y/W/P) ===")
    print(f"\n  {'difficult':>10}  {'N':>4}  {'mean|err|':>9}  {'median|err|':>11}  "
          f"{'max|err|':>8}  {'BAR_err mean':>12}")
    print("  " + "-" * 65)

    for flag, label in [(False, "non-diff"), (True, "difficult")]:
        grp = known[known["is_difficult"] == flag]
        if grp.empty:
            continue
        mean_err = grp["exp_vs_calc_err"].mean()
        med_err = grp["exp_vs_calc_err"].median()
        max_err = grp["exp_vs_calc_err"].max()
        bar_mean = grp["calc_ddG_err"].mean()
        bar_str = f"{bar_mean:.2f}" if not np.isnan(bar_mean) else "   N/A"
        print(
            f"  {label:>10}  {len(grp):>4}  {mean_err:>9.2f}  {med_err:>11.2f}"
            f"  {max_err:>8.2f}  {bar_str:>12}"
        )

    # Also break down by difficulty × n_from_muts
    print("\n  Breakdown by difficulty × n_from_muts:")
    print(f"  {'difficult':>10}  {'n_from_muts':>11}  {'N':>4}  {'mean|err|':>9}  {'BAR_err mean':>12}")
    print("  " + "-" * 55)
    for flag, label in [(False, "non-diff"), (True, "difficult")]:
        grp = known[known["is_difficult"] == flag]
        if grp.empty:
            continue
        for n in sorted(grp["n_from_muts"].unique()):
            sub = grp[grp["n_from_muts"] == n]
            mean_err = sub["exp_vs_calc_err"].mean()
            bar_mean = sub["calc_ddG_err"].mean()
            bar_str = f"{bar_mean:.2f}" if not np.isnan(bar_mean) else "   N/A"
            print(
                f"  {label:>10}  {n:>11}  {len(sub):>4}  {mean_err:>9.2f}  {bar_str:>12}"
            )

def print_flagged_edges(df: pd.DataFrame) -> None:
    """Print edges with potential quality issues."""
    print("\n=== Flagged Edges (potential quality issues) ===")

    # Large exp_vs_calc error (where experimental reference exists)
    known = df.dropna(subset=["exp_vs_calc_err"])
    if not known.empty:
        thresh = known["exp_vs_calc_err"].quantile(0.75)
        bad = known[known["exp_vs_calc_err"] > max(thresh, 2.0)].sort_values(
            "exp_vs_calc_err", ascending=False
        )
        if not bad.empty:
            print(f"\nHigh |calc - expected| (>{max(thresh, 2.0):.2f} kcal/mol):")
            for _, r in bad.iterrows():
                print(
                    f"  {r['system']:6s}  {r['from_mutation']!s:30s} → {r['to_mutation']!s:30s}"
                    f"  calc={r['calc_ddG']:+.2f}  exp={r['expected_ddg']:+.2f}"
                    f"  |err|={r['exp_vs_calc_err']:.2f}"
                    + (f"  BAR_err={r['calc_ddG_err']:.2f}" if not np.isnan(r['calc_ddG_err']) else "")
                )

    # Large cycle closure error
    cyc = df.dropna(subset=["cycle_closure_err"])
    if not cyc.empty:
        thresh_cc = cyc["cycle_closure_err"].quantile(0.75)
        bad_cc = cyc[cyc["cycle_closure_err"] > max(thresh_cc, 1.0)].sort_values(
            "cycle_closure_err", ascending=False
        )
        if not bad_cc.empty:
            print(f"\nHigh cycle closure error (>{max(thresh_cc, 1.0):.2f}):")
            for _, r in bad_cc.iterrows():
                print(
                    f"  {r['system']:6s}  {r['from_mutation']!s:30s} → {r['to_mutation']!s:30s}"
                    f"  calc_ddG={r['calc_ddG']:+.2f}  CC_err={r['cycle_closure_err']:.3f}"
                    + (f"  BAR_err={r['calc_ddG_err']:.2f}" if not np.isnan(r['calc_ddG_err']) else "")
                )

    # High BAR uncertainty
    bar = df.dropna(subset=["calc_ddG_err"])
    if not bar.empty:
        thresh_bar = bar["calc_ddG_err"].quantile(0.75)
        bad_bar = bar[bar["calc_ddG_err"] > max(thresh_bar, 0.5)].sort_values(
            "calc_ddG_err", ascending=False
        )
        if not bad_bar.empty:
            print(f"\nHigh BAR uncertainty (>{max(thresh_bar, 0.5):.2f} kcal/mol):")
            for _, r in bad_bar.iterrows():
                print(
                    f"  {r['system']:6s}  {r['from_mutation']!s:30s} → {r['to_mutation']!s:30s}"
                    f"  calc_ddG={r['calc_ddG']:+.2f}  BAR_err={r['calc_ddG_err']:.2f}"
                    + (f"  CC_err={r['cycle_closure_err']:.3f}" if not np.isnan(r.get('cycle_closure_err', np.nan)) else "")
                    + ("  [difficult]" if r['is_difficult'] else "")
                )

def print_summary_stats(df: pd.DataFrame) -> None:
    """Print per-system and overall summary statistics."""
    print("\n=== Edge Summary Statistics ===")
    print(f"\n{'System':<8} {'N_edges':>7} {'N_cycles':>8} {'|err| mean':>10} "
          f"{'BAR_err mean':>12} {'CC_err mean':>11}")
    print("-" * 60)

    for system in sorted(df["system"].unique()):
        sub = df[df["system"] == system]
        n_edges = len(sub)
        n_cycles = sub["in_cycle"].sum()
        exp_err_mean = sub["exp_vs_calc_err"].mean()
        bar_err_mean = sub["calc_ddG_err"].mean()
        cc_err_mean = sub["cycle_closure_err"].mean()
        exp_str = f"{exp_err_mean:.2f}" if not np.isnan(exp_err_mean) else "  N/A"
        bar_str = f"{bar_err_mean:.2f}" if not np.isnan(bar_err_mean) else "  N/A"
        cc_str = f"{cc_err_mean:.3f}" if not np.isnan(cc_err_mean) else "   N/A"
        print(f"  {system:<6} {n_edges:>7} {n_cycles:>8} {exp_str:>10} {bar_str:>12} {cc_str:>11}")

    # Overall
    print("-" * 60)
    n_total = len(df)
    n_cyc_total = df["in_cycle"].sum()
    oe_mean = df["exp_vs_calc_err"].mean()
    ob_mean = df["calc_ddG_err"].mean()
    oc_mean = df["cycle_closure_err"].mean()
    oe_str = f"{oe_mean:.2f}" if not np.isnan(oe_mean) else "  N/A"
    ob_str = f"{ob_mean:.2f}" if not np.isnan(ob_mean) else "  N/A"
    oc_str = f"{oc_mean:.3f}" if not np.isnan(oc_mean) else "   N/A"
    print(f"  {'TOTAL':<6} {n_total:>7} {n_cyc_total:>8} {oe_str:>10} {ob_str:>12} {oc_str:>11}")

def correlation_report(df: pd.DataFrame) -> None:
    """Report correlation between BAR error and actual prediction error."""
    known = df.dropna(subset=["exp_vs_calc_err", "calc_ddG_err"])
    if len(known) < 3:
        print("\nInsufficient data for BAR_err vs actual_err correlation.")
        return
    corr = np.corrcoef(known["calc_ddG_err"], known["exp_vs_calc_err"])[0, 1]
    print(f"\nCorrelation (BAR_err vs |calc-exp|): r = {corr:.3f}  (N={len(known)})")
    if abs(corr) > 0.5:
        print("  → BAR error is a reasonable proxy for actual prediction error.")
    else:
        print("  → BAR error does NOT reliably predict actual prediction error.")

    cc_known = df.dropna(subset=["exp_vs_calc_err", "cycle_closure_err"])
    if len(cc_known) >= 3:
        corr_cc = np.corrcoef(cc_known["cycle_closure_err"], cc_known["exp_vs_calc_err"])[0, 1]
        print(f"Correlation (CC_err vs |calc-exp|):  r = {corr_cc:.3f}  (N={len(cc_known)})")
        if abs(corr_cc) > 0.5:
            print("  → Cycle closure error is a reasonable proxy for actual prediction error.")
        else:
            print("  → Cycle closure error does NOT reliably predict actual prediction error.")

def make_plots(df: pd.DataFrame, figures_dir: Path) -> None:
    """Generate edge quality plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plots.")
        return

    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. BAR error distribution per system
    fig, ax = plt.subplots(figsize=(10, 5))
    bar_data = df.dropna(subset=["calc_ddG_err"])
    if not bar_data.empty:
        systems = sorted(bar_data["system"].unique())
        data_by_sys = [bar_data[bar_data["system"] == s]["calc_ddG_err"].values for s in systems]
        ax.boxplot(data_by_sys, labels=systems, vert=True)
        ax.set_xlabel("System")
        ax.set_ylabel("BAR uncertainty (kcal/mol)")
        ax.set_title("BAR Error Distribution per System")
        ax.axhline(bar_data["calc_ddG_err"].quantile(0.75), color="red", linestyle="--",
                   alpha=0.6, label="75th percentile")
        ax.legend()
        plt.tight_layout()
        fig.savefig(figures_dir / "bar_err_distribution.png", dpi=150)
    plt.close(fig)

    # 2. Cycle closure error distribution per system
    fig, ax = plt.subplots(figsize=(10, 5))
    cc_data = df.dropna(subset=["cycle_closure_err"])
    if not cc_data.empty:
        systems = sorted(cc_data["system"].unique())
        data_by_sys = [cc_data[cc_data["system"] == s]["cycle_closure_err"].values for s in systems]
        ax.boxplot(data_by_sys, labels=systems, vert=True)
        ax.set_xlabel("System")
        ax.set_ylabel("Cycle closure error (kcal/mol)")
        ax.set_title("Cycle Closure Error Distribution per System")
        plt.tight_layout()
        fig.savefig(figures_dir / "cycle_closure_err_distribution.png", dpi=150)
    plt.close(fig)

    # 3. BAR err vs actual error scatter (where experimental reference available)
    fig, ax = plt.subplots(figsize=(7, 6))
    known = df.dropna(subset=["exp_vs_calc_err", "calc_ddG_err"])
    if not known.empty:
        colors = {"Pathway": "blue", "SingleRef": "orange", "TargetOnly": "green"}
        for cat, grp in known.groupby("category"):
            ax.scatter(grp["calc_ddG_err"], grp["exp_vs_calc_err"],
                       c=colors.get(cat, "gray"), label=cat, alpha=0.7)
        max_val = max(known["calc_ddG_err"].max(), known["exp_vs_calc_err"].max()) * 1.05
        ax.plot([0, max_val], [0, max_val], "k--", alpha=0.3, label="y=x")
        ax.set_xlabel("BAR uncertainty (kcal/mol)")
        ax.set_ylabel("|calc_ddG - expected_ddG| (kcal/mol)")
        ax.set_title("BAR Uncertainty vs Actual Error\n(edges with experimental reference)")
        ax.legend()
        corr = np.corrcoef(known["calc_ddG_err"], known["exp_vs_calc_err"])[0, 1]
        ax.text(0.05, 0.95, f"r = {corr:.3f}", transform=ax.transAxes, va="top")
        plt.tight_layout()
        fig.savefig(figures_dir / "bar_err_vs_actual_err.png", dpi=150)
    plt.close(fig)

    # 4. Cycle closure err vs actual error scatter
    fig, ax = plt.subplots(figsize=(7, 6))
    cc_known = df.dropna(subset=["exp_vs_calc_err", "cycle_closure_err"])
    if not cc_known.empty:
        colors = {"Pathway": "blue", "SingleRef": "orange", "TargetOnly": "green"}
        for cat, grp in cc_known.groupby("category"):
            ax.scatter(grp["cycle_closure_err"], grp["exp_vs_calc_err"],
                       c=colors.get(cat, "gray"), label=cat, alpha=0.7)
        ax.set_xlabel("Cycle closure error")
        ax.set_ylabel("|calc_ddG - expected_ddG| (kcal/mol)")
        ax.set_title("Cycle Closure Error vs Actual Error")
        ax.legend()
        corr_cc = np.corrcoef(cc_known["cycle_closure_err"], cc_known["exp_vs_calc_err"])[0, 1]
        ax.text(0.05, 0.95, f"r = {corr_cc:.3f}", transform=ax.transAxes, va="top")
        plt.tight_layout()
        fig.savefig(figures_dir / "cc_err_vs_actual_err.png", dpi=150)
    plt.close(fig)

    # 5. calc_ddG vs expected_ddG for all edges with experimental reference
    fig, ax = plt.subplots(figsize=(7, 6))
    known2 = df.dropna(subset=["expected_ddg", "calc_ddG"])
    if not known2.empty:
        colors = {"Pathway": "blue", "SingleRef": "orange", "TargetOnly": "green"}
        for cat, grp in known2.groupby("category"):
            ax.scatter(grp["expected_ddg"], grp["calc_ddG"],
                       c=colors.get(cat, "gray"), label=cat, alpha=0.7)
        all_vals = pd.concat([known2["expected_ddg"], known2["calc_ddG"]])
        lo, hi = all_vals.min() - 1, all_vals.max() + 1
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.3, label="y=x")
        ax.set_xlabel("Expected edge ddG (from dGs.json, kcal/mol)")
        ax.set_ylabel("Calculated edge ddG (FEP, kcal/mol)")
        ax.set_title("Calculated vs Expected Edge ddG")
        ax.legend()
        corr_ev = np.corrcoef(known2["expected_ddg"], known2["calc_ddG"])[0, 1]
        ax.text(0.05, 0.95, f"r = {corr_ev:.3f}", transform=ax.transAxes, va="top")
        plt.tight_layout()
        fig.savefig(figures_dir / "calc_vs_expected_edge_ddg.png", dpi=150)
    plt.close(fig)

    # 6. difficulty boxplot
    fig, ax = plt.subplots(figsize=(6, 5))
    diff_data = df.dropna(subset=["exp_vs_calc_err"]).copy()
    if not diff_data.empty:
        groups = []
        labels = []
        for flag, label in [(False, "non-difficult"), (True, "difficult\n(F/Y/W/P)")]:
            sub = diff_data[diff_data["is_difficult"] == flag]["exp_vs_calc_err"].values
            if len(sub) > 0:
                groups.append(sub)
                labels.append(label)
        if groups:
            ax.boxplot(groups, labels=labels, vert=True)
            ax.set_ylabel("|calc_ddG - expected_ddG| (kcal/mol)")
            ax.set_title("FEP Error by Mutation Difficulty")
            plt.tight_layout()
            fig.savefig(figures_dir / "error_by_difficulty.png", dpi=150)
    plt.close(fig)

    # 8. error_by_from_nmuts boxplot
    fig, ax = plt.subplots(figsize=(8, 5))
    known_nmuts = df.dropna(subset=["exp_vs_calc_err"]).copy()
    if not known_nmuts.empty:
        max_n = int(known_nmuts["n_from_muts"].max())
        groups = []
        labels = []
        for n in range(0, max_n + 1):
            sub = known_nmuts[known_nmuts["n_from_muts"] == n]["exp_vs_calc_err"].values
            if len(sub) > 0:
                groups.append(sub)
                labels.append(str(n))
        if groups:
            ax.boxplot(groups, labels=labels, vert=True)
            ax.set_xlabel("Number of mutations in from_node (n_from_muts)")
            ax.set_ylabel("|calc_ddG - expected_ddG| (kcal/mol)")
            ax.set_title("FEP Error vs Reference Structure Mutation Count")
            plt.tight_layout()
            fig.savefig(figures_dir / "error_by_from_nmuts.png", dpi=150)
    plt.close(fig)

    # 9. validated_edges_scatter: calc vs exp, colored by system
    fig, ax = plt.subplots(figsize=(7, 6))
    known_ve = df.dropna(subset=["expected_ddg", "calc_ddG"]).copy()
    if not known_ve.empty:
        systems = sorted(known_ve["system"].unique())
        cmap = plt.get_cmap("tab20")
        sys_colors = {s: cmap(i / max(len(systems) - 1, 1)) for i, s in enumerate(systems)}
        for system, grp in known_ve.groupby("system"):
            ax.scatter(grp["expected_ddg"], grp["calc_ddG"],
                       c=[sys_colors[system]], label=system, alpha=0.8, s=60)
        all_vals = pd.concat([known_ve["expected_ddg"], known_ve["calc_ddG"]])
        lo, hi = all_vals.min() - 1, all_vals.max() + 1
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.3, label="y=x")
        ax.set_xlabel("Expected edge ddG (kcal/mol)")
        ax.set_ylabel("Calculated edge ddG (kcal/mol)")
        ax.set_title("Validated Edges: Calculated vs Expected ddG\n(colored by system)")
        ax.legend(fontsize=7, ncol=2)
        if len(known_ve) >= 2:
            corr_ve = np.corrcoef(known_ve["expected_ddg"], known_ve["calc_ddG"])[0, 1]
            ax.text(0.05, 0.95, f"r = {corr_ve:.3f}", transform=ax.transAxes, va="top")
        plt.tight_layout()
        fig.savefig(figures_dir / "validated_edges_scatter.png", dpi=150)
    plt.close(fig)

    print(f"Figures saved to {figures_dir}/")

def main():
    parser = argparse.ArgumentParser(description="Analyze edge quality in ProtMutMap FEP networks")
    parser.add_argument(
        "--base-dir",
        default=str(Path(__file__).resolve().parents[1]),
        help="experiments/ directory",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: base-dir/results/)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir) if args.output_dir else base_dir / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = base_dir / "figures" / "edge_quality"

    systems_dir = base_dir / "systems"
    if not systems_dir.exists():
        print(f"ERROR: systems directory not found: {systems_dir}")
        return

    all_rows = []
    for sys_dir in sorted(systems_dir.iterdir()):
        if not sys_dir.is_dir():
            continue
        system_name = sys_dir.name
        print(f"Analyzing {system_name}...")
        rows = analyze_system(system_name, sys_dir)
        all_rows.extend(rows)

    if not all_rows:
        print("No edge data found.")
        return

    df = pd.DataFrame(all_rows)

    print_summary_stats(df)
    print_validated_edges(df)
    print_error_by_from_nmuts(df)
    print_error_by_difficulty(df)
    print_flagged_edges(df)
    correlation_report(df)

    # Save CSV
    out_csv = output_dir / "edge_analysis.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved edge analysis to {out_csv}")

    # Generate plots
    if not args.no_plots:
        make_plots(df, figures_dir)

if __name__ == "__main__":
    main()
