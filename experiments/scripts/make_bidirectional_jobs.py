#!/usr/bin/env python3
"""Prepare reverse FEP calculations for charge-changing network edges."""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

from protmutmap.tools.mutations import MutationList

SYSTEMS = {
    "1AO7": ("1AO7_ABC_DE", "1AO7", "1AO7_ABC", "1AO7_DE"),
    "1BJ1": ("1BJ1_HL_VW", "1BJ1", "1BJ1_HL", "1BJ1_VW"),
    "1CHO": ("1CHO_EFG_I", "1CHO", "1CHO_EFG", "1CHO_I"),
    "1R0R": ("1R0R_E_I", "1R0R", "1R0R_E", "1R0R_I"),
    "1MLC": ("1MLC_AB_E", "1MLC", "1MLC_AB", "1MLC_E"),
    "1PPF": ("1PPF_E_I", "1PPF", "1PPF_E", "1PPF_I"),
}

def get_partner_chains(target_id: str) -> dict:
    parts = target_id.split("_")
    return {
        "partner1": list(parts[1]),
        "partner2": list(parts[2]),
    }

def find_mutated_pdb(mutated_pdbs_dir: Path, prefix: str, fs_str: str) -> Optional[Path]:
    """Locate the FASPR-mutated PDB for a given fs_string under {mode}/mutated_pdbs/."""
    candidate = mutated_pdbs_dir / f"{prefix}.{fs_str}.pdb"
    if candidate.exists():
        return candidate
    return None

def run_prep_mutation_fep(
    fepsuite_dir: Path,
    prep_cwd: Path,
    input_pdb: Path,
    reverse_fs: str,
    difficult_nrep: int,
    extreme_nrep: int,
    gmx_path: Optional[str] = None,
    dry_run: bool = False,
) -> bool:
    """Prepare wt_<reverse_fs>/ under prep_cwd from the forward endpoint PDB."""
    faspr_bin = fepsuite_dir / "feprest" / "FASPR" / "FASPR"
    cmd = [
        sys.executable, "-m", "protmutmap.tools.prep_mutation_fep",
        "--fepsuite", str(fepsuite_dir),
        "--faspr", str(faspr_bin),
        "--pdb", str(input_pdb.resolve()),
        "--mutation", reverse_fs,
        "--ff", "amber14sb_OL15_fs1",
        "--difficult-nrep", str(difficult_nrep),
        "--extreme-nrep", str(extreme_nrep),
    ]
    if gmx_path:
        cmd.extend(["--gmx", gmx_path])

    if dry_run:
        print(f"  [DRY] cwd={prep_cwd}")
        print(f"  [DRY] cmd={' '.join(cmd)}")
        return True

    prep_cwd.mkdir(parents=True, exist_ok=True)

    # forcefield symlink at the prep cwd (prep_mutation_fep.py looks up `../<ff>.ff`
    # from inside its generated wt_<rev>/ subdir).
    ff_link = prep_cwd / "amber14sb_OL15_fs1.ff"
    if not ff_link.exists():
        ff_source = fepsuite_dir / "forcefields" / "amber14sb_OL15_fs1.ff"
        if ff_source.exists():
            ff_link.symlink_to(ff_source)

    print(f"  prep: cwd={prep_cwd.name}, mutation={reverse_fs}, input={input_pdb.name}")
    result = subprocess.run(cmd, cwd=str(prep_cwd), capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: prep_mutation_fep.py failed (rc={result.returncode})")
        print(f"  stderr: {result.stderr[-1500:]}")
        return False
    return True

def process_system(
    system: str,
    target_id: str,
    base_pdb_name: str,
    partner1_pdb_name: str,
    partner2_pdb_name: str,
    mutmap_root: Path,
    fepsuite_dir: Path,
    do_prep: bool,
    dry_run: bool,
    gmx_path: Optional[str],
) -> list[dict]:
    """Process one system: enumerate reverse edges and (optionally) run prep."""
    sys_dir = mutmap_root / "experiments" / "systems" / system
    network_dir = sys_dir / "network_charge"
    work_dir = sys_dir / "mutmap_work_large"
    links_path = network_dir / "links.tsv"

    if not links_path.exists():
        print(f"  WARNING: {links_path} not found; skipping {system}")
        return []

    links_df = pd.read_csv(links_path, sep="\t")
    charge_df = links_df[links_df["has_charge_change"] == True].copy()  # noqa: E712
    print(f"=== {system} ({target_id}): {len(charge_df)} charge edges ===")

    if charge_df.empty:
        return []

    partner_chains = get_partner_chains(target_id)
    pdb_prefix_by_mode = {
        "complex": base_pdb_name,
        "partner1": partner1_pdb_name,
        "partner2": partner2_pdb_name,
    }

    job_rows: list[dict] = []

    for _, row in charge_df.iterrows():
        forward_from = row["from_mutation"]
        forward_to = row["to_mutation"]
        is_extreme = bool(row.get("is_extreme", False))
        from_muts = MutationList.from_string(forward_from) if forward_from != "WT" else MutationList()
        to_muts = MutationList.from_string(forward_to) if forward_to != "WT" else MutationList()
        diff_muts = to_muts - from_muts
        if len(diff_muts) == 0:
            continue
        reverse_diff = -diff_muts
        reverse_diff_full_fs = "_".join(reverse_diff.to_fs_mutations())
        forward_diff_fs = "_".join(diff_muts.to_fs_mutations())

        from_state_muts = to_muts
        from_state_full_fs = from_state_muts.to_string(fs=True) if len(from_state_muts) > 0 else "WT"

        nrep = 96 if is_extreme else 64

        for mode in ("complex", "partner1", "partner2"):
            if mode.startswith("partner"):
                filtered_diff = diff_muts.filter_by_chains(partner_chains[mode])
                if len(filtered_diff) == 0:
                    continue
                filtered_reverse = -filtered_diff
                reverse_diff_fs = "_".join(filtered_reverse.to_fs_mutations())
                filtered_from_state = from_state_muts.filter_by_chains(partner_chains[mode])
                from_state_fs = (
                    filtered_from_state.to_string(fs=True)
                    if len(filtered_from_state) > 0
                    else "WT"
                )
            else:
                reverse_diff_fs = reverse_diff_full_fs
                from_state_fs = from_state_full_fs

            mutated_pdbs_dir = work_dir / mode / "mutated_pdbs"
            input_pdb = find_mutated_pdb(mutated_pdbs_dir, pdb_prefix_by_mode[mode], from_state_fs)
            if input_pdb is None:
                print(
                    f"  WARN [{mode}] forward PDB not found for {from_state_fs}: "
                    f"expected {mutated_pdbs_dir}/{pdb_prefix_by_mode[mode]}.{from_state_fs}.pdb"
                )
                continue

            prep_cwd = work_dir / mode / from_state_fs
            target_dir = prep_cwd / f"wt_{reverse_diff_fs}"
            done_marker = prep_cwd / f"{reverse_diff_fs}.done"

            if do_prep and not done_marker.exists():
                ok = run_prep_mutation_fep(
                    fepsuite_dir, prep_cwd, input_pdb, reverse_diff_fs,
                    difficult_nrep=64, extreme_nrep=96,
                    gmx_path=gmx_path, dry_run=dry_run,
                )
                if ok and not dry_run:
                    done_marker.touch()

            job_rows.append({
                "target_dir": str(target_dir.resolve()),
                "system": system,
                "mode": mode,
                "forward_from": forward_from,
                "forward_to": forward_to,
                "forward_diff_fs": forward_diff_fs,
                "reverse_from_state_fs": from_state_fs,
                "reverse_diff_fs": reverse_diff_fs,
                "nrep": nrep,
                "is_difficult": True,
                "is_extreme": is_extreme,
            })

    return job_rows

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mutmap-root",
        default=str(Path(__file__).resolve().parents[2]),
        help="ProtMutMap repo root (default: the repository containing this script)",
    )
    parser.add_argument(
        "--fepsuite-dir",
        default=None,
        help="fepsuite dir (default: $MUTMAP_ROOT/fepsuite)",
    )
    parser.add_argument(
        "--gmx",
        default=None,
        help="gmx binary path (default: $GMXBIN/gmx or PATH lookup)",
    )
    parser.add_argument(
        "--systems",
        default=None,
        help="Comma-separated systems (default: all 6 charge systems)",
    )
    parser.add_argument(
        "--prep", action="store_true",
        help="Run prep_mutation_fep.py for each reverse edge (otherwise only enumerate)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="With --prep, print commands without executing",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Job CSV output path (default: running_large/jobs/mutations_charge_reverse.csv)",
    )
    args = parser.parse_args()

    mutmap_root = Path(args.mutmap_root).resolve()
    fepsuite_dir = Path(args.fepsuite_dir).resolve() if args.fepsuite_dir else mutmap_root / "fepsuite"
    gmx_path = args.gmx
    if gmx_path is None and "GMXBIN" in os.environ:
        gmx_path = os.path.join(os.environ["GMXBIN"], "gmx")
    if gmx_path is None:
        gmx_path = shutil.which("gmx")

    if args.systems:
        requested = [s.strip() for s in args.systems.split(",")]
        unknown = [s for s in requested if s not in SYSTEMS]
        if unknown:
            parser.error(f"Unknown system(s): {', '.join(unknown)}")
        systems_to_process = {k: SYSTEMS[k] for k in requested}
    else:
        systems_to_process = SYSTEMS

    all_rows: list[dict] = []
    for system, (target_id, base_pdb, p1_pdb, p2_pdb) in systems_to_process.items():
        rows = process_system(
            system=system,
            target_id=target_id,
            base_pdb_name=base_pdb,
            partner1_pdb_name=p1_pdb,
            partner2_pdb_name=p2_pdb,
            mutmap_root=mutmap_root,
            fepsuite_dir=fepsuite_dir,
            do_prep=args.prep,
            dry_run=args.dry_run,
            gmx_path=gmx_path,
        )
        all_rows.extend(rows)

    if not all_rows:
        print("No reverse jobs generated.")
        return

    output_csv = Path(args.output_csv) if args.output_csv else (
        mutmap_root / "experiments" / "running_large" / "jobs" / "mutations_charge_reverse.csv"
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(all_rows)
    out_cols = ["target_dir"]
    df[out_cols].to_csv(output_csv, index=False)

    anno_path = output_csv.with_suffix(".annot.tsv")
    df.to_csv(anno_path, sep="\t", index=False)

    print(f"\nWrote {len(df)} reverse-edge job entries to:")
    print(f"  {output_csv}")
    print(f"  {anno_path}  (with edge metadata)")

    for nrep_val, sub in df.groupby("nrep"):
        split_csv = output_csv.with_name(output_csv.stem + f"_nrep{nrep_val}.csv")
        sub[out_cols].to_csv(split_csv, index=False)
        print(f"  {split_csv}  ({len(sub)} entries, NREP={nrep_val})")


if __name__ == "__main__":
    main()
