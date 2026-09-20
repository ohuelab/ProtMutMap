#!/usr/bin/env python3
"""Collect charge-changing FEP results from network_charge and dGs_charge.json."""

import sys
import argparse
import pandas as pd
from pathlib import Path

# Make this script importable from any working directory
_HERE = Path(__file__).resolve()
_SCRIPTS_DIR = _HERE.parent
sys.path.insert(0, str(_SCRIPTS_DIR))

from gather_all_results import (
    gather_system,
    print_metrics_table,
    EXTENDED_METHODS,
)

# charge-changing target mutations (ddG < 0 only)

CHARGE_SYSTEMS = {
    "1AO7": {
        "target_id": "1AO7_ABC_DE",
        "entries": ["AE96M,GE97S,GE98A,RE99Q"],
    },
    "1BJ1": {
        "target_id": "1BJ1_HL_VW",
        "entries": ["HH101Y,YH103R,SH105T"],
    },
    "1CHO": {
        "target_id": "1CHO_EFG_I",
        "entries": ["LI15M,NI33D,SI48N"],
    },
    "1R0R": {
        "target_id": "1R0R_E_I",
        "entries": ["LI13M,NI31D,SI46N"],
    },
    "1MLC": {
        "target_id": "1MLC_AB_E",
        "entries": ["NA92A,SB57V,TB58D"],
    },
    "1PPF": {
        "target_id": "1PPF_E_I",
        "entries": ["LI18M,NI36D,SI51N"],
    },
}

# All charge-changing systems are "Pathway"-style (intermediate ddGs available
# in dGs_charge.json) since they have single + sub-combination experimental data.
CHARGE_SYSTEM_CATEGORY = {name: "Pathway" for name in CHARGE_SYSTEMS}

def main():
    parser = argparse.ArgumentParser(
        description="Gather charge-changing FEP results"
    )
    parser.add_argument(
        "--base-dir",
        default=str(Path(__file__).resolve().parents[1]),
        help="experiments/ directory",
    )
    parser.add_argument(
        "--work-dir",
        default="mutmap_work_large",
        help="Work directory name (default: mutmap_work_large)",
    )
    parser.add_argument(
        "--network-dir-name",
        default="network_charge",
        help="Network directory name under each system dir",
    )
    parser.add_argument(
        "--dgs-filename",
        default="dGs_charge.json",
        help="Experimental ddG JSON filename",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: base-dir/results_large_charge)",
    )
    parser.add_argument(
        "--systems",
        default=None,
        help="Comma-separated systems (default: all 6 charge systems)",
    )
    parser.add_argument(
        "--time-ps",
        type=float,
        default=None,
        help="If set, read per-leg BAR estimates from bar_time_series.csv at "
             "this cutoff (ps) instead of bar1.log. The default output dir "
             "gets a _<N>ns suffix (e.g. results_large_charge → "
             "results_large_charge_4ns) so the full-time results are kept.",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    time_ps = args.time_ps
    cutoff_suffix = "" if time_ps is None else f"_{int(round(time_ps / 1000))}ns"
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else base_dir / f"results_large_charge{cutoff_suffix}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.systems:
        requested = [s.strip() for s in args.systems.split(",")]
        unknown = [s for s in requested if s not in CHARGE_SYSTEMS]
        if unknown:
            parser.error(f"Unknown system(s): {', '.join(unknown)}")
        systems = {k: CHARGE_SYSTEMS[k] for k in requested}
    else:
        systems = CHARGE_SYSTEMS

    all_target_rows = []
    all_intermediate_rows = []
    all_diagnostics_rows = []
    all_bidir_rows = []

    for system_name, system_info in systems.items():
        print(f"\n{'='*60}")
        print(f"Processing: {system_name}")
        target_df, intermediate_df, diag_df, bidir_df = gather_system(
            system_name=system_name,
            system_info=system_info,
            base_dir=base_dir,
            work_dir_name=args.work_dir,
            extra_networks=None,
            network_dir_name=args.network_dir_name,
            dgs_filename=args.dgs_filename,
            system_category_map=CHARGE_SYSTEM_CATEGORY,
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
    intermediate_df = (
        pd.concat(all_intermediate_rows, ignore_index=True)
        if all_intermediate_rows else pd.DataFrame()
    )
    all_df = (
        pd.concat([target_df, intermediate_df], ignore_index=True)
        if not intermediate_df.empty else target_df
    )

    target_csv = output_dir / "target_results.csv"
    target_df.to_csv(target_csv, index=False)
    print(f"\nSaved target results to {target_csv}")
    print(target_df.to_string())

    if not intermediate_df.empty:
        inter_csv = output_dir / "intermediate_results.csv"
        intermediate_df.to_csv(inter_csv, index=False)
        print(f"\nSaved intermediate results to {inter_csv}")

    all_csv = output_dir / "all_results.csv"
    all_df.to_csv(all_csv, index=False)
    print(f"\nSaved combined results to {all_csv}")

    if all_diagnostics_rows:
        diag_df_all = pd.concat(all_diagnostics_rows, ignore_index=True)
        diag_csv = output_dir / "edge_diagnostics.csv"
        diag_df_all.to_csv(diag_csv, index=False)
        print(f"\nSaved edge cycle diagnostics to {diag_csv}")

    if all_bidir_rows:
        bidir_df_all = pd.concat(all_bidir_rows, ignore_index=True)
        bidir_csv = output_dir / "edge_bidirectional.csv"
        bidir_df_all.to_csv(bidir_csv, index=False)
        print(f"\nSaved bidirectional charge-edge table to {bidir_csv}")

    print("\n" + "=" * 60)
    metrics_rows = []
    metrics_rows += print_metrics_table(target_df, "Metrics — Target entries only:", EXTENDED_METHODS)
    if not intermediate_df.empty:
        metrics_rows += print_metrics_table(all_df, "Metrics — All data points (targets + intermediates):", EXTENDED_METHODS)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_csv = output_dir / "metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"\nSaved metrics to {metrics_csv}")

if __name__ == "__main__":
    main()
