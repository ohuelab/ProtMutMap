#!/usr/bin/env python3
"""Prepare FEP calculations from network edges and starting structures."""

import os
import subprocess
import pandas as pd
import argparse
import logging
from pathlib import Path
from typing import Dict, Set, Optional, List, Tuple
import sys
import shutil

from .tools.mutations import MutationList
from .tools.crystal_picker import pick_nearest_crystal
from .mutant_preparer import MutationPreparer

def _crystal_safe_dirname(crystal_mut: str) -> str:
    """Sanitize a mutation_str for use as a directory name."""
    return crystal_mut.replace(",", "_")

class FEPSuitePreparer:
    """Prepare structures and FEP inputs for network edges."""

    def __init__(self,
                 link_file: str,
                 target: str,
                 base_target_dir: str,
                 node_file: Optional[str] = None,
                 partners: List[str] = None,
                 fepsuite_dir: str = "fepsuite",
                 faspr: Optional[str] = None,
                 gmx: Optional[str] = None,
                 override: bool = False,
                 complex_pdb: Optional[str] = None,
                 partner1_pdb: Optional[str] = None,
                 partner2_pdb: Optional[str] = None,
                 verbose: bool = False,
                 default_nrep: Optional[int] = None,
                 difficult_nrep: Optional[int] = None,
                 charge_nrep: Optional[int] = None,
                 extreme_nrep: Optional[int] = None,
                 keep_nonmutated_rotamers: bool = False,
                 crystal_pdbs: Optional[Dict[str, Dict[str, str]]] = None):
        """
        Initialize the FEPSuitePreparer.

        Args:
            link_file: Path to the link file containing mutation connections
            target: Target identifier (e.g., 1AO7_ABC_DE)
            base_target_dir: Base directory for output files
            node_file: Optional path to node file with pre-generated PDB files
            partners: List of partners to process (default: ["partner1"])
            fepsuite_dir: Path to FEP suite directory
            faspr: Path to FASPR binary (auto-detected if not provided)
            gmx: Path to GROMACS gmx binary (auto-detected if not provided)
            override: Whether to override existing files
            complex_pdb: Wild-type PDB file for complex
            partner1_pdb: Wild-type PDB file for partner1
            partner2_pdb: Wild-type PDB file for partner2
            verbose: Enable verbose logging
            default_nrep: Default number of replicas (passed to prep_mutation_fep.py)
            difficult_nrep: Number of replicas for difficult/bulky mutations (F/Y/W/P)
            charge_nrep: Number of replicas for charge-changing mutations
            extreme_nrep: Number of replicas for extreme mutations
            keep_nonmutated_rotamers: Keep original rotamers for non-mutated residues (FASPR mutation-site-only mode)
        """
        self.link_file = Path(link_file)
        self.target = target
        self.base_target_dir = Path(base_target_dir)
        self.node_file = Path(node_file) if node_file else None
        self.partners = partners or ["partner1"]
        self.fepsuite_dir = Path(fepsuite_dir).resolve()
        self.faspr = Path(faspr).resolve() if faspr else None
        self.gmx = gmx
        self.override = override
        self.complex_pdb = Path(complex_pdb) if complex_pdb else None
        self.partner1_pdb = Path(partner1_pdb) if partner1_pdb else None
        self.partner2_pdb = Path(partner2_pdb) if partner2_pdb else None
        self.default_nrep = default_nrep
        self.difficult_nrep = difficult_nrep
        self.charge_nrep = charge_nrep
        self.extreme_nrep = extreme_nrep
        self.keep_nonmutated_rotamers = keep_nonmutated_rotamers
        # Optional mutation_str -> per-leg crystal PDB paths.
        self.crystal_pdbs: Dict[str, Dict[str, str]] = crystal_pdbs or {}

        self.logger = self._setup_logging(verbose)

        self.links_df: Optional[pd.DataFrame] = None
        self.pdb_sources: Dict[str, Dict[str, str]] = {}
        # mutation_str -> chosen source crystal mutation_str (per node)
        self.node_source_crystal: Dict[str, str] = {}
        self.calculation_files: List[str] = []
        self.failed_files: List[str] = []

        self._validate_initialization()

    def _setup_logging(self, verbose: bool) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG if verbose else logging.INFO)
        return logger

    def _validate_initialization(self) -> None:
        """Validate initialization parameters."""
        if not self.link_file.exists():
            raise FileNotFoundError(f"Link file not found: {self.link_file}")

        if not self.faspr:
            self.faspr = self.fepsuite_dir / "feprest" / "FASPR" / "FASPR"

        if not self.faspr.exists():
            raise ValueError(f"FASPR binary not found: {self.faspr}")

        self.logger.info(f"Initialized FEPSuitePreparer for target: {self.target}")
        self.logger.info(f"Link file: {self.link_file}")
        self.logger.info(f"Partners: {self.partners}")

    def validate_input_data(self) -> Dict[str, Dict[str, str]]:
        """
        Validate input data and determine PDB source strategy.

        Returns:
            Dictionary mapping mutations to PDB file paths for each partner

        Raises:
            ValueError: If required data is missing
        """
        pdb_sources = {}

        if self.node_file and self.node_file.exists():
            try:
                nodes_df = pd.read_csv(self.node_file, sep='\t')
                required_columns = ['mutations']

                partner_columns = {}
                if 'complex' in self.partners:
                    if 'complex_pdb_file' in nodes_df.columns:
                        partner_columns['complex'] = 'complex_pdb_file'
                    else:
                        raise ValueError("complex in partners but complex_pdb_file column missing from node file")

                for partner in self.partners:
                    if partner.startswith('partner'):
                        col_name = f"{partner}_pdb_file"
                        if col_name in nodes_df.columns:
                            partner_columns[partner] = col_name
                        else:
                            raise ValueError(f"{partner} in partners but {col_name} column missing from node file")

                if partner_columns:
                    self.logger.info("Using pre-generated PDB files from node file")
                    for _, row in nodes_df.iterrows():
                        mutation = row['mutations']
                        pdb_sources[mutation] = {}
                        for partner, col_name in partner_columns.items():
                            pdb_sources[mutation][partner] = row[col_name]
                    return pdb_sources

            except Exception as e:
                self.logger.warning(f"Could not read node file: {e}")

        self.logger.info("Using wild-type PDB arguments for on-demand generation")

        if not self.complex_pdb:
            raise ValueError("--complex-pdb is required when node file is not available or incomplete")

        for partner in self.partners:
            if partner == 'partner1' and not self.partner1_pdb:
                raise ValueError("--partner1-pdb is required when partner1 is in --partners")
            elif partner == 'partner2' and not self.partner2_pdb:
                raise ValueError("--partner2-pdb is required when partner2 is in --partners")

        return {}  # Empty dict indicates we need to generate PDBs


    def _resolve_source_crystal(self, from_mutation: str) -> Optional[str]:
        """Return the source crystal mutation_str for a from-state node.

        If ``from_mutation`` is itself a registered crystal, it returns itself.
        Otherwise the nearest crystal by symmetric-set difference is picked
        (with alphabetical tie-break). Returns ``None`` when no crystals are
        configured.
        """
        if not self.crystal_pdbs:
            return None
        if from_mutation in self.node_source_crystal:
            return self.node_source_crystal[from_mutation]
        if from_mutation in self.crystal_pdbs:
            self.node_source_crystal[from_mutation] = from_mutation
            return from_mutation
        picked = pick_nearest_crystal(from_mutation, self.crystal_pdbs.keys())
        if picked is None:
            return None
        self.node_source_crystal[from_mutation] = picked
        return picked


    def extract_unique_mutations(self, links_df: pd.DataFrame) -> Set[str]:
        """
        Extract unique mutations from links DataFrame.

        Args:
            links_df: DataFrame containing mutation links

        Returns:
            Set of unique mutation strings
        """
        unique_mutations = set()
        unique_mutations.update(links_df['from_mutation'].unique())
        unique_mutations.update(links_df['to_mutation'].unique())
        self.logger.info(f"Extracted {len(unique_mutations)} unique mutations")
        return unique_mutations

    def _get_default_pdb_for_leg(self, partner: str) -> Optional[str]:
        """Return the single-PDB input for the given leg, if any."""
        if partner == "complex":
            return str(self.complex_pdb) if self.complex_pdb else None
        if partner == "partner1":
            return str(self.partner1_pdb) if self.partner1_pdb else None
        if partner == "partner2":
            return str(self.partner2_pdb) if self.partner2_pdb else None
        return None

    def _get_source_pdb_for_leg(
        self,
        source_crystal_mut: Optional[str],
        partner: str,
    ) -> Optional[str]:
        """Pick the PDB file representing ``source_crystal_mut`` for ``partner``.

        Falls back to the single-PDB input if the per-crystal entry
        is missing."""
        if source_crystal_mut and source_crystal_mut in self.crystal_pdbs:
            paths = self.crystal_pdbs[source_crystal_mut]
            if partner in paths:
                return paths[partner]
        return self._get_default_pdb_for_leg(partner)

    def _get_or_create_preparer(
        self,
        source_crystal_mut: Optional[str],
        partner: str,
        cache: Dict[Tuple[Optional[str], str], MutationPreparer],
    ) -> Optional[MutationPreparer]:
        """Lazily build a ``MutationPreparer`` rooted at ``source_crystal_mut``.

        With multiple crystals each gets its own work_dir so FASPR outputs
        don't collide. When no crystals are configured this falls back to
        the per-leg work directory."""
        key = (source_crystal_mut, partner)
        if key in cache:
            return cache[key]
        pdb_path = self._get_source_pdb_for_leg(source_crystal_mut, partner)
        if pdb_path is None:
            cache[key] = None  # type: ignore[assignment]
            return None
        if source_crystal_mut and self.crystal_pdbs:
            sub = _crystal_safe_dirname(source_crystal_mut)
            work_dir = self.base_target_dir / partner / f"crystal_{sub}"
        else:
            work_dir = self.base_target_dir / partner
        preparer = MutationPreparer(
            input_pdb_file=pdb_path,
            work_dir=work_dir,
            faspr_bin=self.faspr,
            override=self.override,
        )
        cache[key] = preparer
        return preparer

    def prepare_pdb_files(self, unique_mutations: Set[str]) -> Dict[str, Dict[str, str]]:
        """
        Prepare PDB files for all unique mutations using MutationPreparer.

        Args:
            unique_mutations: Set of unique mutation strings

        Returns:
            Dictionary mapping mutations to PDB file paths for each partner
        """
        pdb_sources: Dict[str, Dict[str, str]] = {}

        partner_chains_dict: Dict[str, List[str]] = {}
        if "_" in self.target:
            parts = self.target.split("_")
            if len(parts) >= 3:
                partner_chains_dict = {
                    "partner1": list(parts[1]),
                    "partner2": list(parts[2])
                }

        # Cache of (source_crystal, partner) -> MutationPreparer
        preparer_cache: Dict[Tuple[Optional[str], str], MutationPreparer] = {}

        active_partners = []
        if 'complex' in self.partners:
            active_partners.append('complex')
        for partner in self.partners:
            if partner.startswith('partner'):
                active_partners.append(partner)


        total_mutations = len(unique_mutations)
        for i, mutation in enumerate(unique_mutations, 1):
            self.logger.info(f"Preparing PDB files for mutation {i}/{total_mutations}: {mutation}")
            pdb_sources[mutation] = {}

            if mutation == "WT":
                muts = MutationList()
            else:
                muts = MutationList.from_string(mutation)

            source_crystal_mut = self._resolve_source_crystal(mutation)

            for partner in active_partners:
                preparer = self._get_or_create_preparer(
                    source_crystal_mut, partner, preparer_cache
                )
                if preparer is None:
                    self.logger.debug(
                        f"No PDB available for {mutation} in {partner}; skipping."
                    )
                    continue

                try:
                    if partner.startswith("partner") and partner_chains_dict:
                        partner_muts = muts.filter_by_chains(
                            partner_chains_dict[partner]
                        )
                        if len(partner_muts) == 0 and mutation != "WT":
                            self.logger.debug(
                                f"Skipping {partner}: mutations don't affect partner chains"
                            )
                            continue
                    else:
                        partner_muts = muts

                    # Compute FASPR build mutations = target - source.
                    # When no crystal source is configured, the source is implicitly
                    # the WT PDB without mutations.
                    if source_crystal_mut and source_crystal_mut != "WT":
                        source_muts_all = MutationList.from_string(source_crystal_mut)
                    else:
                        source_muts_all = MutationList()

                    if partner.startswith("partner") and partner_chains_dict:
                        source_muts_leg = source_muts_all.filter_by_chains(
                            partner_chains_dict[partner]
                        )
                    else:
                        source_muts_leg = source_muts_all

                    faspr_muts = partner_muts - source_muts_leg

                    preparer.run(faspr_muts)

                    target_stem = Path(preparer.input_pdb_file).stem
                    if len(faspr_muts) == 0:
                        fs_mutations_str = "WT"
                    else:
                        fs_mutations_str = "_".join(faspr_muts.to_fs_mutations())
                    output_file = (
                        preparer.mutant_output_dir / f"{target_stem}.{fs_mutations_str}.pdb"
                    )
                    pdb_sources[mutation][partner] = str(output_file)

                except Exception as e:
                    self.logger.error(
                        f"Failed to prepare PDB for {mutation} in {partner}: {e}"
                    )
                    raise

        return pdb_sources

    def process_fep_calculations(self, links_df: pd.DataFrame, pdb_sources: Dict[str, Dict[str, str]]) -> None:
        """
        Process FEP calculations using links.

        Args:
            links_df: DataFrame containing mutation links
            pdb_sources: Dictionary mapping mutations to PDB file paths
        """
        partner_chains_dict = {}
        if "_" in self.target:
            parts = self.target.split("_")
            if len(parts) >= 3:
                partner_chains_dict = {
                    "partner1": list(parts[1]),
                    "partner2": list(parts[2])
                }

        total_edges = len(links_df)
        self.logger.info(f"Processing {total_edges} FEP calculation edges")

        for i, (_, row) in enumerate(links_df.iterrows(), 1):
            from_mutation = row['from_mutation']
            to_mutation = row['to_mutation']
            mutation_diff = row['mutation_diff']

            self.logger.info(f"Processing edge {i}/{total_edges}: {from_mutation} -> {to_mutation} (diff: {mutation_diff})")

            try:
                self._process_single_edge(row, pdb_sources, partner_chains_dict)
            except Exception as e:
                self.logger.error(f"Failed to process edge {from_mutation} -> {to_mutation}: {e}")
                continue

        self.logger.info("FEP calculation processing completed!")
        self.logger.info(f"Successful calculations: {len(self.calculation_files)}")
        self.logger.info(f"Failed calculations: {len(self.failed_files)}")

    def _process_single_edge(self, row: pd.Series, pdb_sources: Dict[str, Dict[str, str]],
                           partner_chains_dict: Dict[str, List[str]]) -> None:
        """Process a single FEP calculation edge."""
        from_mutation = row['from_mutation']
        to_mutation = row['to_mutation']


        from_muts = MutationList.from_string(from_mutation) if from_mutation != "WT" else MutationList()
        to_muts = MutationList.from_string(to_mutation) if to_mutation != "WT" else MutationList()
        diff_muts = to_muts - from_muts

        if len(diff_muts) == 0:
            self.logger.warning(f"No mutations to calculate for link {from_mutation} to {to_mutation}")
            return

        diff_fs_mutations = diff_muts.to_fs_mutations()
        diff_fs_mutations_str = "_".join(diff_fs_mutations)


        for mode in self.partners:
            try:
                self._process_partner_mode(from_mutation, mode, pdb_sources, diff_muts,
                                         from_muts, diff_fs_mutations_str,
                                         partner_chains_dict, to_mutation)
            except Exception as e:
                self.logger.error(f"Failed to process partner mode {mode}: {e}")
                continue

    def _process_partner_mode(self, from_mutation: str, mode: str,
                            pdb_sources: Dict[str, Dict[str, str]],
                            diff_muts: MutationList, from_muts: MutationList,
                            diff_fs_mutations_str: str,
                            partner_chains_dict: Dict[str, List[str]],
                            to_mutation: Optional[str] = None) -> None:
        """Process FEP calculation for a specific partner mode."""
        if mode.startswith("partner") and partner_chains_dict:
            partner_chains = partner_chains_dict[mode]
            filtered_from = from_muts.filter_by_chains(partner_chains)
            from_fs_mutations_str = filtered_from.to_string(fs=True) if len(filtered_from) > 0 else "WT"

            filtered_diff = diff_muts.filter_by_chains(partner_chains)
            if len(filtered_diff) == 0:
                self.logger.debug(f"Skipping {mode}: mutations don't affect partner chains")
                return
            partner_diff_fs_str = "_".join(filtered_diff.to_fs_mutations())

            if from_mutation in pdb_sources and mode in pdb_sources[from_mutation]:
                input_pdb = Path(pdb_sources[from_mutation][mode])
            elif len(filtered_from) == 0:
                # from_mutation doesn't affect this partner — use WT partner PDB
                if "WT" in pdb_sources and mode in pdb_sources["WT"]:
                    input_pdb = Path(pdb_sources["WT"][mode])
                else:
                    self.logger.debug(f"No WT PDB found for {mode} - skipping")
                    return
            else:
                self.logger.debug(f"No PDB file found for {from_mutation} in mode {mode} - skipping")
                return
        else:
            from_fs_mutations_str = from_muts.to_string(fs=True) if len(from_muts) > 0 else "WT"
            partner_diff_fs_str = diff_fs_mutations_str
            if from_mutation in pdb_sources and mode in pdb_sources[from_mutation]:
                input_pdb = Path(pdb_sources[from_mutation][mode])
            else:
                self.logger.debug(f"No PDB file found for {from_mutation} in mode {mode} - skipping (normal for mutations not affecting this partner)")
                return

        target_dir = self.base_target_dir / mode / from_fs_mutations_str


        if (target_dir / f"{partner_diff_fs_str}.done").exists() and not self.override:
            self.calculation_files.append(str((target_dir / f"wt_{partner_diff_fs_str}").resolve()))
            self.logger.debug(f"Already done: {target_dir / f'{partner_diff_fs_str}.done'}")
            return

        target_dir.mkdir(parents=True, exist_ok=True)

        self._create_forcefield_symlink(target_dir)

        # Run FEP preparation — the mutation string is interpreted against
        # input_pdb's own residue numbering (prep_mutation_fep.py looks up
        # the residue directly in the PDB).
        success = self._run_fep_preparation(input_pdb, partner_diff_fs_str, target_dir)

        if success:
            (target_dir / f"{partner_diff_fs_str}.done").touch()
            self.calculation_files.append(str((target_dir / f"wt_{partner_diff_fs_str}").resolve()))
            self.logger.debug(f"Success: {target_dir / f'{partner_diff_fs_str}.done'}")
        else:
            self.failed_files.append(str((target_dir / f"wt_{partner_diff_fs_str}").resolve()))

    def _create_forcefield_symlink(self, target_dir: Path) -> None:
        """Create forcefield symlink in target directory."""
        ff_link = target_dir / "amber14sb_OL15_fs1.ff"
        if not ff_link.exists():
            ff_source = self.fepsuite_dir / "forcefields" / "amber14sb_OL15_fs1.ff"
            if ff_source.exists():
                ff_link.symlink_to(ff_source)
            else:
                self.logger.warning(f"Forcefield file not found: {ff_source}")

    def _run_fep_preparation(self, input_pdb: Path, diff_fs_mutations_str: str,
                           target_dir: Path) -> bool:
        """Run FEP preparation command and return success status.

        ``diff_fs_mutations_str`` is interpreted against ``input_pdb``'s own
        residue numbering (``prep_mutation_fep.py`` looks up the residue
        directly in the PDB).
        """
        cmd = [
            sys.executable,
            "-m", "protmutmap.tools.prep_mutation_fep",
            "--faspr",
            str(self.faspr),
            "--fepsuite", str(self.fepsuite_dir),
            "--pdb", str(input_pdb.resolve()),
            "--mutation", diff_fs_mutations_str,
            "--ff", "amber14sb_OL15_fs1"
        ]

        if self.default_nrep is not None:
            cmd.extend(["--default-nrep", str(self.default_nrep)])
        if self.difficult_nrep is not None:
            cmd.extend(["--difficult-nrep", str(self.difficult_nrep)])
        if self.charge_nrep is not None:
            cmd.extend(["--charge-nrep", str(self.charge_nrep)])
        if self.extreme_nrep is not None:
            cmd.extend(["--extreme-nrep", str(self.extreme_nrep)])

        # Keep original rotamers for non-mutated residues (mutation-site-only FASPR mode)
        if self.keep_nonmutated_rotamers:
            cmd.append("--keep-nonmutated-rotamers")

        gmx_path = self._get_gmx_path()
        if gmx_path and os.path.exists(gmx_path):
            cmd.extend(["--gmx", gmx_path])
        elif gmx_path:
            self.logger.warning(f"gmx path {gmx_path} not found")
        else:
            self.logger.warning("gmx not found. Please set --gmx, GMXBIN environment variable, or add gmx to PATH.")

        self.logger.debug(f"Running: {' '.join(cmd)}")
        self.logger.debug(f"Working directory: {target_dir}")

        ret = subprocess.run(cmd, cwd=str(target_dir), capture_output=True, text=True)

        if ret.returncode != 0:
            self.logger.error(f"FEP preparation failed with return code: {ret.returncode}")
            if ret.stderr:
                self.logger.error(f"stderr: {ret.stderr}")
            if ret.stdout:
                self.logger.error(f"stdout: {ret.stdout}")
            return False

        return True

    def _get_gmx_path(self) -> Optional[str]:
        """Get GROMACS gmx binary path."""
        if self.gmx:
            return self.gmx
        elif "GMXBIN" in os.environ:
            return os.environ["GMXBIN"] + "/gmx"
        else:
            return shutil.which("gmx")

    def run(self) -> Tuple[List[str], List[str]]:
        """
        Run the complete FEP suite preparation workflow.

        Returns:
            Tuple of (successful_calculations, failed_calculations)
        """
        try:
            self.logger.info(f"Reading links file: {self.link_file}")
            self.links_df = pd.read_csv(self.link_file, sep='\t')
            self.logger.info(f"Found {len(self.links_df)} mutation links")

            self.pdb_sources = self.validate_input_data()

            if not self.pdb_sources:
                unique_mutations = self.extract_unique_mutations(self.links_df)
                self.pdb_sources = self.prepare_pdb_files(unique_mutations)

            self.process_fep_calculations(self.links_df, self.pdb_sources)

            return self.calculation_files, self.failed_files

        except Exception as e:
            self.logger.error(f"FEP suite preparation failed: {e}")
            raise

    def get_summary(self) -> Dict[str, any]:
        """Get a summary of the processing results."""
        return {
            "target": self.target,
            "total_links": len(self.links_df) if self.links_df is not None else 0,
            "successful_calculations": len(self.calculation_files),
            "failed_calculations": len(self.failed_files),
            "calculation_files": self.calculation_files,
            "failed_files": self.failed_files
        }

def main():
    """Prepare FEP inputs from command-line arguments."""
    parser = argparse.ArgumentParser(description="Process FEP calculations from TSV link files")

    parser.add_argument("--link-file", required=True, help="Path to link file")
    parser.add_argument("--target", required=True, help="Target identifier (e.g., 1AO7_ABC_DE)")
    parser.add_argument("--base-target-dir", required=True, help="Base target directory path")

    parser.add_argument("--node-file", help="Optional path to node file")
    parser.add_argument("--partners", nargs="*", default=["complex", "partner1", "partner2"], help="List of partners")
    parser.add_argument("--fepsuite-dir", default="fepsuite", help="Path to FEP suite directory")
    parser.add_argument("--faspr", default=None, help="Path to FASPR binary (auto-detected if not provided)")
    parser.add_argument("--gmx", default=None, help="Path to GROMACS gmx binary (auto-detected if not provided)")
    parser.add_argument("--override", action="store_true", help="Override existing files")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    parser.add_argument("--complex-pdb", help="Wild-type PDB file for complex")
    parser.add_argument("--partner1-pdb", help="Wild-type PDB file for partner1")
    parser.add_argument("--partner2-pdb", help="Wild-type PDB file for partner2")

    parser.add_argument("--default-nrep", type=int, default=None,
                        help="Default number of replicas (overrides prep_mutation_fep.py default)")
    parser.add_argument("--difficult-nrep", type=int, default=None,
                        help="Number of replicas for difficult/bulky mutations (F/Y/W/P)")
    parser.add_argument("--charge-nrep", type=int, default=None,
                        help="Number of replicas for charge-changing (non-bulky) mutations")
    parser.add_argument("--extreme-nrep", type=int, default=None,
                        help="Number of replicas for extreme mutations")
    parser.add_argument("--keep-nonmutated-rotamers", action="store_true",
                        help="Keep original rotamers for non-mutated residues (mutation-site-only FASPR mode)")

    parser.add_argument("--crystal-pdbs", default=None,
                        help="Optional TSV mapping mutation_str to per-leg "
                             "crystal PDB paths (cols: mutation_str, complex_pdb, "
                             "partner1_pdb, partner2_pdb). When supplied, each node "
                             "starts from its registered crystal (or FASPR from the "
                             "nearest one).")

    args = parser.parse_args()


    crystal_pdbs = None
    if args.crystal_pdbs:
        from .cli import load_crystal_pdbs
        crystal_pdbs = load_crystal_pdbs(args.crystal_pdbs)

    try:

        preparer = FEPSuitePreparer(
            link_file=args.link_file,
            target=args.target,
            base_target_dir=args.base_target_dir,
            node_file=args.node_file,
            partners=args.partners,
            fepsuite_dir=args.fepsuite_dir,
            faspr=args.faspr,
            gmx=args.gmx,
            override=args.override,
            complex_pdb=args.complex_pdb,
            partner1_pdb=args.partner1_pdb,
            partner2_pdb=args.partner2_pdb,
            verbose=args.verbose,
            default_nrep=args.default_nrep,
            difficult_nrep=args.difficult_nrep,
            charge_nrep=args.charge_nrep,
            extreme_nrep=args.extreme_nrep,
            keep_nonmutated_rotamers=args.keep_nonmutated_rotamers,
            crystal_pdbs=crystal_pdbs,
        )

        calculation_files, failed_files = preparer.run()

        print(f"\ncalculation_files ({len(calculation_files)}):")
        for file in calculation_files:
            print(file)
        print(f"\nfailed_files ({len(failed_files)}):")
        for file in failed_files:
            print(file)

        print("\nFEP calculation processing completed!")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
