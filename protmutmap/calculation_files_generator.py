#!/usr/bin/env python3
"""Generate FEP calculation paths from a table of network edges."""

import pandas as pd
from pathlib import Path
from typing import List, Dict, Set, Optional
import logging
from .tools.mutations import MutationList

def generate_calculation_files(
    link_file: str,
    target: str,
    base_target_dir: str,
    partners: Optional[List[str]] = None,
    check_existing: bool = True,
    logger: Optional[logging.Logger] = None
) -> List[str]:
    """
    Generate calculation file paths from link file and target information.

    This function extracts the core logic from FEPSuitePreparer to generate
    the list of calculation files that would be created, without actually
    running the FEP preparation workflow.

    Args:
        link_file: Path to the link file containing mutation connections
        target: Target identifier (e.g., 1AO7_ABC_DE)
        base_target_dir: Base directory for output files
        partners: List of partners to process (default: ["partner1"])
        check_existing: Whether to check for existing .done files
        logger: Optional logger instance

    Returns:
        List of calculation file paths that would be generated

    Raises:
        FileNotFoundError: If link file doesn't exist
        ValueError: If link file format is invalid
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    link_file_path = Path(link_file)
    if not link_file_path.exists():
        raise FileNotFoundError(f"Link file not found: {link_file}")

    base_target_dir_path = Path(base_target_dir)
    partners = partners or ["partner1"]

    logger.info(f"Generating calculation files for target: {target}")
    logger.info(f"Link file: {link_file}")
    logger.info(f"Partners: {partners}")

    try:
        links_df = pd.read_csv(link_file_path, sep='\t')
        logger.info(f"Found {len(links_df)} mutation links")
    except Exception as e:
        raise ValueError(f"Failed to read link file: {e}")

    required_columns = ['from_mutation', 'to_mutation', 'mutation_diff']
    missing_columns = [col for col in required_columns if col not in links_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in link file: {missing_columns}")

    partner_chains_dict = {}
    if "_" in target:
        parts = target.split("_")
        if len(parts) >= 3:
            partner_chains_dict = {
                "partner1": list(parts[1]),
                "partner2": list(parts[2])
            }

    calculation_files = []

    total_edges = len(links_df)
    logger.info(f"Processing {total_edges} FEP calculation edges")

    for i, (_, row) in enumerate(links_df.iterrows(), 1):
        from_mutation = row['from_mutation']
        to_mutation = row['to_mutation']

        logger.debug(f"Processing edge {i}/{total_edges}: {from_mutation} -> {to_mutation}")

        try:
            edge_files = _process_single_edge(
                row, base_target_dir_path, partners, partner_chains_dict,
                check_existing, logger
            )
            calculation_files.extend(edge_files)
        except Exception as e:
            logger.error(f"Failed to process edge {from_mutation} -> {to_mutation}: {e}")
            continue

    logger.info(f"Generated {len(calculation_files)} calculation file paths")
    return calculation_files

def _process_single_edge(
    row: pd.Series,
    base_target_dir: Path,
    partners: List[str],
    partner_chains_dict: Dict[str, List[str]],
    check_existing: bool,
    logger: logging.Logger
) -> List[str]:
    """Process a single FEP calculation edge and return calculation file paths."""
    from_mutation = row['from_mutation']
    to_mutation = row['to_mutation']


    from_muts = MutationList.from_string(from_mutation) if from_mutation != "WT" else MutationList()
    to_muts = MutationList.from_string(to_mutation) if to_mutation != "WT" else MutationList()
    diff_muts = to_muts - from_muts

    if len(diff_muts) == 0:
        logger.warning(f"No mutations to calculate for link {from_mutation} to {to_mutation}")
        return []

    from_fs_mutations_str = from_muts.to_string(fs=True) if len(from_muts) > 0 else "WT"
    diff_fs_mutations = diff_muts.to_fs_mutations()
    diff_fs_mutations_str = "_".join(diff_fs_mutations)

    edge_files = []


    for mode in partners:
        try:
            file_path = _process_partner_mode(
                mode, base_target_dir, diff_muts, from_fs_mutations_str,
                diff_fs_mutations_str, partner_chains_dict, check_existing, logger
            )
            if file_path:
                edge_files.append(file_path)
        except Exception as e:
            logger.error(f"Failed to process partner mode {mode}: {e}")
            continue

    return edge_files

def _process_partner_mode(
    mode: str,
    base_target_dir: Path,
    diff_muts: MutationList,
    from_fs_mutations_str: str,
    diff_fs_mutations_str: str,
    partner_chains_dict: Dict[str, List[str]],
    check_existing: bool,
    logger: logging.Logger
) -> Optional[str]:
    """Process FEP calculation for a specific partner mode and return file path."""

    if mode.startswith("partner") and partner_chains_dict:
        mut_chains = diff_muts.get_mutation_chains()
        is_hit = False
        for chain in mut_chains:
            if chain in partner_chains_dict[mode]:
                is_hit = True
                break
        if not is_hit:
            logger.debug(f"Skipping {mode}: mutations don't affect partner chains")
            return None

    target_dir = base_target_dir / mode / from_fs_mutations_str

    if check_existing and (target_dir / f"{diff_fs_mutations_str}.done").exists():
        logger.debug(f"Already done: {target_dir / f'{diff_fs_mutations_str}.done'}")

    calculation_file = target_dir / f"wt_{diff_fs_mutations_str}"
    return str(calculation_file.resolve())

def get_unique_mutations_from_links(link_file: str) -> Set[str]:
    """
    Extract unique mutations from a links file.

    Args:
        link_file: Path to the link file containing mutation connections

    Returns:
        Set of unique mutation strings

    Raises:
        FileNotFoundError: If link file doesn't exist
        ValueError: If link file format is invalid
    """
    link_file_path = Path(link_file)
    if not link_file_path.exists():
        raise FileNotFoundError(f"Link file not found: {link_file}")

    try:
        links_df = pd.read_csv(link_file_path, sep='\t')
    except Exception as e:
        raise ValueError(f"Failed to read link file: {e}")

    required_columns = ['from_mutation', 'to_mutation']
    missing_columns = [col for col in required_columns if col not in links_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in link file: {missing_columns}")

    unique_mutations = set()
    unique_mutations.update(links_df['from_mutation'].unique())
    unique_mutations.update(links_df['to_mutation'].unique())

    return unique_mutations

def main():
    """List calculation directories for a network's edges."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Generate calculation file paths from link files")
    parser.add_argument("--link-file", required=True, help="Path to link file")
    parser.add_argument("--target", required=True, help="Target identifier (e.g., 1AO7_ABC_DE)")
    parser.add_argument("--base-target-dir", required=True, help="Base target directory path")
    parser.add_argument("--partners", nargs="*", default=["partner1"], help="List of partners")
    parser.add_argument("--no-check-existing", action="store_true", help="Don't check for existing .done files")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)

    try:
        calculation_files = generate_calculation_files(
            link_file=args.link_file,
            target=args.target,
            base_target_dir=args.base_target_dir,
            partners=args.partners,
            check_existing=not args.no_check_existing,
            logger=logger
        )

        print(f"\ncalculation_files ({len(calculation_files)}):")
        for file in calculation_files:
            print(file)

        print(f"\nGenerated {len(calculation_files)} calculation file paths")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
