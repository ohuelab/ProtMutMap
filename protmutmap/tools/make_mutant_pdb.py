#!/usr/bin/env python3
import os
import argparse
import subprocess
import shutil
from typing import Sequence, Optional

from .prep_mutation_fep import PDBInfoSummary, update_mutinfo, parse_pdb
from .mutation import Mutation, parse_mutations

def write_faspr_seq(ddir: str, basepdbinfo: PDBInfoSummary, mutations: Sequence[Mutation]):
    """Generate FASPR seq.txt file."""
    os.makedirs(ddir, exist_ok=True)
    with open(os.path.join(ddir, "seq.txt"), "w") as ofh:
        curchain = None
        reschainmap = basepdbinfo.res_chain_dic
        totmut = 0
        for (c, r) in basepdbinfo.order:
            if curchain != c:
                if curchain is not None:
                    print("", file=ofh)
                chainname = c if c.strip() != "" else "X"
                print(f">{chainname}", file=ofh)
                curchain = c
            res = reschainmap[r][c]
            mutated = False
            for m in mutations:
                if m.resid == r and (m.chain is None or m.chain == c):
                    if m.before_res is not None and m.before_res != res:
                        raise RuntimeError(f"Mutation ({m}) is not compatible with chain '{c}' resid {r}")
                    res = m.after_res
                    mutated = True
                    totmut += 1
                    break
            if (res == "C") and (not mutated):
                res = "c"
            print(res, end="", file=ofh)
        print("", file=ofh)
        if totmut != len(mutations):
            raise RuntimeError(f"Not all mutations are consumed, requested: {mutations}")

def make_mutant_pdb(pdb_path: str,
                    mutation_str: str,
                    faspr_bin: str,
                    output_path: str,
                    work_dir: str,
                    seed: Optional[int] = None) -> None:
    """
    Generate variant-A PDB from WT PDB with specified mutations.

    Args:
        pdb_path: Input WT base PDB file path
        mutation_str: Mutation string for variant-A (e.g., 'H:35A_H:50V'). If empty, generates repacked WT
        faspr_bin: Path to FASPR binary
        output_path: Output PDB file path (variant-A)
        work_dir: Working directory for intermediate files
        seed: Optional random seed for FASPR

    Raises:
        RuntimeError: If FASPR execution fails
        FileNotFoundError: If input files are not found
    """
    # Parse PDB and mutations
    base = parse_pdb(pdb_path)
    if mutation_str.strip():
        muts = parse_mutations(mutation_str)
        muts = update_mutinfo(muts, base)
    else:
        muts = []

    # Generate FASPR sequence file
    write_faspr_seq(work_dir, base, muts)

    # Prepare FASPR command
    completed_pdb = os.path.join(work_dir, "completed.pdb")
    seq_file = os.path.join(work_dir, "seq.txt")

    cmd = [faspr_bin, "-i", pdb_path, "-o", completed_pdb, "-s", seq_file]
    if seed is not None:
        cmd += ["-seed", str(seed)]

    print(f"Running: {' '.join(cmd)}")
    subprocess.check_call(cmd)

    # Verify output and copy to final location
    if (not os.path.exists(completed_pdb)) or (os.path.getmtime(completed_pdb) < os.path.getmtime(seq_file)):
        raise RuntimeError("FASPR run failed")

    shutil.copy2(completed_pdb, output_path)
    print(f"Done. Wrote variant-A PDB to: {output_path}")

def main():
    """Command-line interface for make_mutant_pdb function."""
    p = argparse.ArgumentParser(description="Generate variant-A PDB from WT PDB with specified mutations")
    p.add_argument("--pdb", required=True, help="Input WT base PDB file")
    p.add_argument("--mut", default="", help="Mutation string for variant-A (e.g., H:35A_H:50V). If empty, generates repacked WT")
    p.add_argument("--faspr", required=True, help="Path to FASPR binary")
    p.add_argument("--out", required=True, help="Output PDB file (variant-A)")
    p.add_argument("--workdir", default="variantA_build", help="Working directory for intermediate files")
    p.add_argument("--seed", type=int, default=None, help="Random seed for FASPR (optional)")
    args = p.parse_args()

    make_mutant_pdb(
        pdb_path=args.pdb,
        mutation_str=args.mut,
        faspr_bin=args.faspr,
        output_path=args.out,
        work_dir=args.workdir,
        seed=args.seed
    )

if __name__ == "__main__":
    main()
