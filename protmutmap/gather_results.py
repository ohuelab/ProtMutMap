import argparse
import logging
import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional

from protmutmap.tools import MutationList
from protmutmap.wcc.main import wcc_from_dataframe


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Prefer wt_<diff>_<suffix> directories in this order, then wt_<diff>.
# Shared with direct callers of _process_single_edge so box corrections and
# convergence analysis resolve the same calculation directories.
LEG_DIR_VARIANTS: List[str] = []

def resolve_leg_dir(leg_dir: Path, variants: Optional[List[str]] = None) -> Path:
    """Return the preferred existing variant of ``leg_dir``.

    Tries ``<leg_dir>_<suffix>`` for each suffix in ``variants`` (defaulting to
    the module-global :data:`LEG_DIR_VARIANTS`) in order and returns the first
    that is a directory; otherwise returns ``leg_dir`` unchanged.
    """
    leg_dir = Path(leg_dir)
    for suffix in (LEG_DIR_VARIANTS if variants is None else variants):
        candidate = leg_dir.parent / f"{leg_dir.name}_{suffix}"
        if candidate.is_dir():
            return candidate
    return leg_dir

def leg_dir_variant(leg_dir: Path) -> Optional[str]:
    """Return the active variant suffix of ``leg_dir``, or None if unsuffixed."""
    name = Path(leg_dir).name
    for suffix in LEG_DIR_VARIANTS:
        if name.endswith(f"_{suffix}"):
            return suffix
    return None

def _process_partner_mode(
    mode: str,
    base_target_dir: Path,
    from_muts: MutationList,
    diff_muts: MutationList,
    diff_fs_mutations_str: str,
    partner_chains_dict: Dict[str, List[str]],
) -> Optional[str]:
    """
    Process FEP calculation for a specific partner mode and return file path.

    Args:
        mode: Partner mode ('complex', 'partner1', or 'partner2')
        base_target_dir: Base directory containing calculation results
        from_muts: Starting mutations
        diff_muts: Differential mutations to apply
        diff_fs_mutations_str: FEPSuite format mutation string
        partner_chains_dict: Dictionary mapping partner modes to chain lists

    Returns:
        Path to calculation file, or None if mutations don't affect this partner
    """

    if mode.startswith("partner") and partner_chains_dict:
        mut_chains = diff_muts.get_mutation_chains()
        affects_partner = False
        for chain in mut_chains:
            if chain in partner_chains_dict[mode]:
                affects_partner = True
                break
        if not affects_partner:
            logger.debug(f"Skipping {mode}: mutations don't affect chains {partner_chains_dict[mode]}")
            return None

    if mode.startswith("partner") and partner_chains_dict:
        filtered_from_muts = from_muts.filter_by_chains(partner_chains_dict[mode])
        from_fs_mutations_str = filtered_from_muts.to_string(fs=True) if len(filtered_from_muts) > 0 else "WT"
        filtered_diff_muts = diff_muts.filter_by_chains(partner_chains_dict[mode])
        partner_diff_fs_str = "_".join(filtered_diff_muts.to_fs_mutations())
    else:
        from_fs_mutations_str = from_muts.to_string(fs=True) if len(from_muts) > 0 else "WT"
        partner_diff_fs_str = diff_fs_mutations_str

    target_dir = base_target_dir / mode / from_fs_mutations_str

    calculation_file = resolve_leg_dir(target_dir / f"wt_{partner_diff_fs_str}")
    return str(calculation_file.resolve())

def _process_single_edge(
    row: pd.Series,
    base_target_dir: Path,
    partner_chains_dict: Dict[str, List[str]],
) -> Dict[str, str]:
    """
    Process a single FEP calculation edge and return calculation file paths.

    Args:
        row: DataFrame row containing 'from_mutation' and 'to_mutation'
        base_target_dir: Base directory containing calculation results
        partner_chains_dict: Dictionary mapping partner modes to chain lists

    Returns:
        Dictionary mapping mode names to calculation file paths
    """
    from_mutation = row['from_mutation']
    to_mutation = row['to_mutation']


    from_muts = MutationList.from_string(from_mutation) if from_mutation != "WT" else MutationList()
    to_muts = MutationList.from_string(to_mutation) if to_mutation != "WT" else MutationList()
    diff_muts = to_muts - from_muts

    if len(diff_muts) == 0:
        logger.warning(f"No mutations to calculate for edge {from_mutation} → {to_mutation}")
        return {}

    diff_fs_mutations = diff_muts.to_fs_mutations()
    diff_fs_mutations_str = "_".join(diff_fs_mutations)
    edge_files = {}


    for mode in ["complex", "partner1", "partner2"]:
        try:
            file_path = _process_partner_mode(
                mode, base_target_dir, from_muts, diff_muts,
                diff_fs_mutations_str, partner_chains_dict
            )
            if file_path:
                edge_files[mode] = file_path
                pass  # File found
        except Exception as e:
            logger.error(f"  {mode}: Failed to process - {e}")
            continue

    return edge_files

def read_bar1_results(file_path: Path) -> tuple[float, float]:
    """Read BAR1 calculation results from log file."""
    try:
        with open(file_path, "r") as f:
            lines = f.read().splitlines()
        if not lines:
            raise ValueError(f"Empty file: {file_path}")
        # Search from the end for the final "BAR <dG> <err>" line
        # (some bar1.log files have trailing lines like "Wrote ./bar_time_series.csv")
        bar_line = None
        for line in reversed(lines):
            parts = line.split()
            if len(parts) >= 3 and parts[0] == "BAR":
                try:
                    float(parts[1])
                    float(parts[2])
                    bar_line = parts
                    break
                except ValueError:
                    continue
        if bar_line is None:
            raise ValueError(f"Invalid format in {file_path}: no valid BAR line found")
        return float(bar_line[1]), float(bar_line[2])
    except FileNotFoundError:
        logger.error(f"BAR1 file not found: {file_path}")
        raise
    except (ValueError, IndexError) as e:
        logger.error(f"Failed to parse BAR1 file {file_path}: {e}")
        raise ValueError(f"Invalid BAR1 file format: {file_path}") from e

_BAR_TIME_SERIES_CSV = "bar_time_series.csv"
_TIME_PS_TOL = 1e-3
# Tolerance (ps) for "bar1.log's effective tmax matches the requested cutoff".
# fepsuite's prodrun records frames every ~10 ps and the recorded tmax is
# typically ~100 ps over the nominal end time (e.g. 4100 ps for a 4 ns run).
# 200 ps accommodates that without allowing 4 ns vs 8 ns confusion.
_BAR1_TMAX_TOL_PS = 200.0
_RE_BAR_SPLIT_TMAX = re.compile(r"^BAR SPLIT\s+[\d.]+-([\d.]+):")

def _bar1_tmax_ps(bar1_path: Path) -> float | None:
    """Return the last SPLIT window end-time from a bar1.log, or None.

    The SPLIT-block endpoints span the latter half of the prod-run, so the
    final ``BAR SPLIT t0-t1:`` line's t1 is a tight upper bound on the
    sampled tmax. Used to decide whether a leg's bar1.log is consistent
    with a requested cutoff.
    """
    if not bar1_path.exists():
        return None
    last_t1: float | None = None
    try:
        with open(bar1_path) as fp:
            for line in fp:
                m = _RE_BAR_SPLIT_TMAX.match(line)
                if m:
                    last_t1 = float(m.group(1))
    except OSError:
        return None
    return last_t1

def read_bar_at_time(
    leg_dir: Path,
    time_ps: float | None,
    *,
    allow_bar1_fallback: bool = True,
) -> tuple[float, float]:
    """Return (dG_kcal, dG_err_kcal) for a FEP leg at the requested cutoff.

    ``time_ps=None`` reads ``bar1.log`` (full-simulation final BAR estimate)
    via :func:`read_bar1_results`.

    When ``time_ps`` is supplied, the resolution order is:

      1. ``<leg_dir>/bar_time_series.csv`` with a row whose ``max_time_ps``
         matches ``time_ps`` (tolerance ``_TIME_PS_TOL`` ps) AND
         ``status == "ok"``.
      2. **Fallback** (only when ``allow_bar1_fallback=True``, the default):
         if the CSV is missing/has no matching row and ``<leg_dir>/bar1.log``
         exists with an effective tmax (last BAR SPLIT t1) within
         ``_BAR1_TMAX_TOL_PS`` of ``time_ps``, accept the bar1.log final
         value. Pass ``allow_bar1_fallback=False`` to require the CSV path
         exclusively.

    No nearest-row fallback is taken in the CSV path; mixing time
    windows across complex/partner1/partner2 would silently corrupt
    downstream ddG estimates.
    """
    leg_dir = Path(leg_dir)
    if time_ps is None:
        return read_bar1_results(leg_dir / "bar1.log")

    csv_path = leg_dir / _BAR_TIME_SERIES_CSV
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            raise FileNotFoundError(f"Failed to read {csv_path}: {e}") from e

        if "max_time_ps" not in df.columns or "status" not in df.columns:
            raise FileNotFoundError(
                f"{csv_path} missing required columns max_time_ps/status"
            )

        matches = df[
            (np.isclose(df["max_time_ps"].astype(float), float(time_ps), atol=_TIME_PS_TOL))
            & (df["status"].astype(str) == "ok")
        ]
        if not matches.empty:
            row = matches.iloc[0]
            return float(row["dG_kcal"]), float(row["dG_err_kcal"])
        # CSV exists but no matching status=ok row → fall through to bar1.log
        # fallback (a leg may have CSV with all non-ok statuses if the
        # original prod was shorter than every requested cutoff).

    if not allow_bar1_fallback:
        raise FileNotFoundError(
            f"No status=ok row at max_time_ps={time_ps} in {csv_path} "
            f"(bar1.log fallback disabled)"
        )

    # bar1.log fallback: only accept when bar1.log's effective tmax (last
    # BAR SPLIT t1) is consistent with the requested cutoff. This covers
    # cleaned legs whose reps.tar.gz is gone but whose bar1.log already
    # captures the cutoff we want.
    bar1_path = leg_dir / "bar1.log"
    tmax = _bar1_tmax_ps(bar1_path)
    if tmax is not None and abs(tmax - float(time_ps)) <= _BAR1_TMAX_TOL_PS:
        return read_bar1_results(bar1_path)

    raise FileNotFoundError(
        f"No status=ok row at max_time_ps={time_ps} in {csv_path} "
        f"(bar1.log tmax={tmax})"
    )

def gather_results(
    links_df: pd.DataFrame,
    base_target_dir: Path,
    partner_chains_dict: Dict[str, List[str]],
    *,
    time_ps: Optional[float] = None,
    allow_bar1_fallback: bool = True,
) -> pd.DataFrame:
    """Gather FEP calculation results for all edges in the mutation graph.

    Args:
        links_df: DataFrame containing mutation edges (from_mutation, to_mutation)
        base_target_dir: Base directory containing calculation results
        partner_chains_dict: Dictionary mapping partner modes to chain lists
        time_ps: If None (default), reads bar1.log (full-simulation BAR).
            If a float, reads bar_time_series.csv and uses the row at this
            cutoff (status=="ok" only; mismatches → edge marked incomplete).
        allow_bar1_fallback: forwarded to :func:`read_bar_at_time`. Default
            ``True`` keeps the bar1.log fallback (used when the CSV
            row is unavailable but bar1.log's effective tmax matches the
            requested cutoff). Pass ``False`` to require the CSV path
            exclusively — phase2 4 ns analysis sets this to forbid silent
            bar1.log mixing.

    Returns:
        DataFrame with calculation results including dG values and errors.
        Edges with any expected partner leg missing or unparseable at the
        requested cutoff have calc_ddG=NaN; partner legs filtered out by
        _process_partner_mode (mutation does not affect that partner's
        chains) are treated as 0, matching the established convention."""
    total_edges = len(links_df)

    calc_data_list = []
    files_found_count = {"complex": 0, "partner1": 0, "partner2": 0}
    files_missing_count = {"complex": 0, "partner1": 0, "partner2": 0}

    for i, row in links_df.iterrows():
        edge_files = _process_single_edge(
            row,
            base_target_dir,
            partner_chains_dict=partner_chains_dict,
        )

        calc_data = {**row}
        # Unchanged partners contribute zero; affected partners require a result.
        expected_modes = set(edge_files.keys())
        got_modes: set[str] = set()

        for mode, file_path in edge_files.items():
            leg_dir = Path(file_path)

            try:
                dg_val, dg_err = read_bar_at_time(
                    leg_dir, time_ps,
                    allow_bar1_fallback=allow_bar1_fallback,
                )
                files_found_count[mode] += 1
                got_modes.add(mode)

                if mode == "partner1":
                    calc_data["partner1_dG"] = dg_val
                    calc_data["partner1_dG_err"] = dg_err
                elif mode == "partner2":
                    calc_data["partner2_dG"] = dg_val
                    calc_data["partner2_dG_err"] = dg_err
                elif mode == "complex":
                    calc_data["complex_dG"] = dg_val
                    calc_data["complex_dG_err"] = dg_err
            except (FileNotFoundError, ValueError) as e:
                logger.error(f"  Failed to read {mode} results at time_ps={time_ps}: {e}")
                files_missing_count[mode] += 1
                continue

        # Record resolved directories when suffix preferences are active.
        if LEG_DIR_VARIANTS:
            calc_data["leg_variants"] = ";".join(
                f"{mode}:{leg_dir_variant(Path(path)) or 'base'}"
                for mode, path in sorted(edge_files.items())
            )

        # Missing required legs yield NaN, never a partial binding estimate.
        if "complex" not in expected_modes or got_modes != expected_modes:
            calc_data["calc_ddG"] = np.nan
            calc_data["calc_ddG_err"] = np.nan
        else:
            calc_data["calc_ddG"] = (
                calc_data["complex_dG"]
                - calc_data.get("partner1_dG", 0)
                - calc_data.get("partner2_dG", 0)
            )
            calc_data["calc_ddG_err"] = np.sqrt(
                calc_data.get("complex_dG_err", 0) ** 2
                + calc_data.get("partner1_dG_err", 0) ** 2
                + calc_data.get("partner2_dG_err", 0) ** 2
            )

        calc_data_list.append(calc_data)

    calc_result_df = pd.DataFrame(calc_data_list)
    return calc_result_df



if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Gather FEP calculation results and perform weighted cycle closure analysis"
    )
    parser.add_argument("--target-id", type=str, required=True,
                        help="Target ID in format: PDB_CHAIN1_CHAIN2 (e.g., 1A4Y_A_B)")
    parser.add_argument("--node-file", type=str, required=True,
                        help="Path to nodes TSV file")
    parser.add_argument("--link-file", type=str, required=True,
                        help="Path to links TSV file")
    parser.add_argument("--use-uncertainty", action="store_true",
                        help="Use uncertainty in WCC calculation")
    parser.add_argument("--base-target-dir", type=str, required=True,
                        help="Base directory containing calculation results")
    parser.add_argument("--target-mutations", type=str, default=None,
                        help="Specific target mutations to report")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for results")
    parser.add_argument("--time-ps", type=float, default=None,
                        help="If set, read the BAR estimate at this cutoff "
                             "(ps) from <leg>/bar_time_series.csv instead of "
                             "the full-simulation bar1.log. Output filenames "
                             "get a _t{cutoff} suffix to avoid clobbering "
                             "the full-time results.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_parts = args.target_id.split("_")
    if len(target_parts) != 3:
        logger.error(f"Invalid target-id format: {args.target_id}. Expected format: PDB_CHAIN1_CHAIN2")
        raise ValueError(f"Invalid target-id format: {args.target_id}")

    target_pdb = target_parts[0]
    partner1_chains = target_parts[1]
    partner2_chains = target_parts[2]


    partner_chains_dict = {
        "partner1": list(partner1_chains),
        "partner2": list(partner2_chains)
    }

    nodes_df = pd.read_csv(args.node_file, sep="\t", index_col=0)
    links_df = pd.read_csv(args.link_file, sep="\t")
    base_target_dir = Path(args.base_target_dir)

    calc_result_df = gather_results(
        links_df, base_target_dir, partner_chains_dict, time_ps=args.time_ps
    )


    if args.use_uncertainty:
        logger.info("Using uncertainty in WCC calculation")
        result = wcc_from_dataframe(
            calc_result_df[["from_mutation", "to_mutation", "calc_ddG", "calc_ddG_err"]],
            ref_mol="WT",
            mol1_col='from_mutation',
            mol2_col='to_mutation',
            ddg_col='calc_ddG',
            uncertainty_col="calc_ddG_err"
        )
    else:
        result = wcc_from_dataframe(
            calc_result_df[["from_mutation", "to_mutation", "calc_ddG"]],
            ref_mol="WT",
            mol1_col='from_mutation',
            mol2_col='to_mutation',
            ddg_col='calc_ddG'
        )

    wcc_calc_dGs = pd.Series(result["energies"][0], index=result["molecules"])
    wcc_path_independent_errors = pd.Series(
        result["path_independent_errors"],
        index=result["molecules"]
    ).astype(float)
    wcc_path_dependent_errors = pd.Series(
        result["path_dependent_errors"],
        index=result["molecules"]
    ).astype(float)


    nodes_df.loc[wcc_calc_dGs.index, "wcc_calc_dG"] = wcc_calc_dGs
    nodes_df.loc[wcc_calc_dGs.index, "wcc_path_dependent_error"] = wcc_path_dependent_errors
    nodes_df.loc[wcc_calc_dGs.index, "wcc_path_independent_error"] = wcc_path_independent_errors

    calc_result_df["wcc_calc_ddG"] = (
        calc_result_df["to_mutation"].apply(lambda x: nodes_df.loc[x, "wcc_calc_dG"])
        - calc_result_df["from_mutation"].apply(lambda x: nodes_df.loc[x, "wcc_calc_dG"])
    )
    calc_result_df["wcc_calc_ddG_err"] = np.sqrt(
        calc_result_df["to_mutation"].apply(lambda x: nodes_df.loc[x, "wcc_path_dependent_error"]) ** 2
        + calc_result_df["from_mutation"].apply(lambda x: nodes_df.loc[x, "wcc_path_dependent_error"]) ** 2
    )

    if args.target_mutations is not None:
        exp_dG = nodes_df.loc[args.target_mutations, "dG"]
        wcc_dG = nodes_df.loc[args.target_mutations, "wcc_calc_dG"]
        wcc_dG_err = nodes_df.loc[args.target_mutations, "wcc_path_dependent_error"]

        logger.info(f"Mutation: {args.target_mutations}")
        logger.info(f"Experimental dG: {exp_dG:.3f}")
        logger.info(f"WCC calculated dG: {wcc_dG:.3f} ± {wcc_dG_err:.3f}")
        logger.info(f"Difference: {abs(exp_dG - wcc_dG):.3f}")

    suffix = "" if args.time_ps is None else f"_t{int(round(args.time_ps))}"
    nodes_output = output_dir / f"wcc_nodes{suffix}.csv"
    links_output = output_dir / f"wcc_links{suffix}.csv"

    nodes_df.to_csv(nodes_output, sep="\t")
    logger.info(f"Saved nodes results to: {nodes_output}")

    calc_result_df.to_csv(links_output, sep="\t", index=False)
    logger.info(f"Saved links results to: {links_output}")
