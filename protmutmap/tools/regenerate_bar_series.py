"""Backfill bar_time_series.csv for FEP legs by replaying bar_deltae.py.

A FEP leg is identified by a directory containing both ``reps.tar.gz``
(the per-replica deltae.xvg archive produced by fepsuite) and the standard
pipeline outputs. For each such leg, this script invokes the vendored
``fepsuite/feprest/bar_deltae.py`` with ``--time-cutoffs`` so that a per-leg
``bar_time_series.csv`` is generated alongside the existing ``bar1.log``.

Usage:
    python -m protmutmap.tools.regenerate_bar_series \\
        --base-dirs fepsuite_work fepsuite_work_rest \\
        --time-cutoffs 1000,2000,4000,8000 \\
        --jobs 8

Existing CSVs are skipped unless ``--force`` is passed or the recorded
cutoff set differs from ``--time-cutoffs``.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import re
import subprocess
import sys
import tarfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Sequence

logger = logging.getLogger("regenerate_bar_series")

# Resolve once at import time: fepsuite ships in-tree under the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BAR_DELTAE = _REPO_ROOT / "fepsuite" / "feprest" / "bar_deltae.py"
DEFAULT_TIME_CUTOFFS = "1000,2000,4000,8000"
DEFAULT_TEMP = 300.0
TIME_SERIES_CSV = "bar_time_series.csv"
# bar_deltae.py imports pymbar 3.x and calls MBAR.getFreeEnergyDifferences,
# which the mutmap env's pymbar 4.x dropped. The fepsuite env pins pymbar
# 3.0.3 and is the only one that runs bar_deltae correctly out of the box.
# Set FEPSUITE_PYTHON to that interpreter; it falls back to the current one.
_DEFAULT_FEPSUITE_PYTHON = os.environ.get("FEPSUITE_PYTHON", sys.executable)

# Leg discovery

def find_legs(base_dirs: Sequence[Path]) -> list[Path]:
    """Return sorted unique list of directories containing FEP deltae data.

    A leg is identified by EITHER:
      - ``<leg>/reps.tar.gz`` (archived form), OR
      - ``<leg>/prodrun/rep*/deltae.xvg`` (un-archived; reps.tar.gz never
        produced or removed but raw replicas still present).

    The tar archive is preferred when both exist (cheaper for bar_deltae.py
    to read a single file), but raw-prodrun-only legs are NOT skipped.

    Uses GNU ``find`` via subprocess — Python's ``Path.rglob`` is dramatically
    slower than ``find`` on this filesystem (10s vs 15+ minutes on a 481-leg
    tree with deep symlink chains).
    """
    legs: set[Path] = set()
    for base in base_dirs:
        base = Path(base)
        if not base.exists():
            logger.warning("Base dir not found: %s", base)
            continue
        try:
            tar_proc = subprocess.run(
                ["find", "-L", str(base), "-name", "reps.tar.gz"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, check=False,
            )
            xvg_proc = subprocess.run(
                ["find", "-L", str(base), "-path", "*/prodrun/rep0/deltae.xvg"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, check=False,
            )
        except FileNotFoundError:
            logger.warning("`find` not available; falling back to Python rglob (slow)")
            for tar in base.rglob("reps.tar.gz"):
                legs.add(tar.parent.resolve())
            for xvg in base.rglob("prodrun/rep0/deltae.xvg"):
                legs.add(xvg.parents[2].resolve())
            continue
        for line in tar_proc.stdout.splitlines():
            p = Path(line.strip())
            if p.is_file():
                legs.add(p.parent.resolve())
        for line in xvg_proc.stdout.splitlines():
            p = Path(line.strip())
            if p.is_file():
                # <leg>/prodrun/rep0/deltae.xvg → leg is parents[2]
                legs.add(p.parents[2].resolve())
    return sorted(legs)

def _has_reps_tar(leg: Path) -> bool:
    """True iff <leg>/reps.tar.gz exists and is non-empty (>200 bytes)."""
    tar = leg / "reps.tar.gz"
    try:
        return tar.is_file() and tar.stat().st_size > 200
    except OSError:
        return False

def _detect_nsim_from_prodrun(leg: Path) -> int | None:
    """Count prodrun/rep*/deltae.xvg files to infer nsim."""
    prodrun = leg / "prodrun"
    if not prodrun.is_dir():
        return None
    indices: set[int] = set()
    for entry in prodrun.iterdir():
        m = re.match(r"rep(\d+)$", entry.name)
        if m and (entry / "deltae.xvg").is_file():
            indices.add(int(m.group(1)))
    if not indices:
        return None
    return max(indices) + 1

def load_leg_list(path: Path) -> list[Path]:
    """Read a newline-separated list of leg directories (one per line)."""
    legs: list[Path] = []
    with open(path) as fp:
        for line in fp:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            legs.append(Path(line).resolve())
    return legs

# Per-leg metadata detection

_NREP_RE = re.compile(r"^\s*(?:export\s+)?NREP=(\d+)")
_REF_T_RE = re.compile(r"^\s*ref[-_]t\s*=\s*([0-9.eE+-]+)")

def detect_nsim(leg: Path) -> Optional[int]:
    """Auto-detect the number of replicas for a leg.

    Resolution order:
      1. ``para_conf.zsh`` (per-leg or any parent up to base_target_dir) — NREP=...
      2. Count ``reps/deltae_rep*.xvg`` entries inside reps.tar.gz.
      3. Count ``prodrun/rep*/deltae.xvg`` on disk.
    """
    # 1. para_conf.zsh — search leg and parents (up to 4 levels up)
    p: Path = leg
    for _ in range(5):
        candidate = p / "para_conf.zsh"
        if candidate.is_file():
            try:
                with open(candidate) as fp:
                    for line in fp:
                        m = _NREP_RE.match(line)
                        if m:
                            return int(m.group(1))
            except OSError:
                pass
        p = p.parent

    # 2. Count from the tar archive
    tar_path = leg / "reps.tar.gz"
    if _has_reps_tar(leg):
        try:
            with tarfile.open(tar_path, "r:gz") as tf:
                rep_indices = set()
                for name in tf.getnames():
                    m = re.search(r"deltae_rep(\d+)\.xvg$", name)
                    if m:
                        rep_indices.add(int(m.group(1)))
            if rep_indices:
                return max(rep_indices) + 1
        except (tarfile.TarError, OSError) as e:
            logger.warning("Failed to inspect %s: %s", tar_path, e)

    # 3. Fall back to counting raw prodrun replicas.
    return _detect_nsim_from_prodrun(leg)

def detect_temp(leg: Path, default: float) -> float:
    """Auto-detect simulation temperature from prodrun/rep0/run.mdp.

    Falls back to ``default`` if the mdp is missing or unparseable.
    """
    mdp = leg / "prodrun" / "rep0" / "run.mdp"
    if mdp.is_file():
        try:
            with open(mdp) as fp:
                for line in fp:
                    if line.lstrip().startswith(";"):
                        continue
                    key, _, value = line.partition("=")
                    if key.strip().lower().replace("_", "-") == "ref-t":
                        # ref-t may have multiple temperatures per group; take first
                        first = value.strip().split()[0]
                        return float(first)
        except (OSError, ValueError) as e:
            logger.debug("Could not parse ref-t in %s: %s", mdp, e)
    return float(default)

# CSV freshness check

def existing_csv_matches(csv_path: Path, requested_cutoffs: list[float]) -> bool:
    """True iff ``csv_path`` exists and lists exactly the requested cutoff set.

    Comparison ignores ordering and tolerates float rounding (within 1e-3 ps).
    Rows are not validated for status; the goal here is "did the previous run
    already produce all requested cutoffs?".
    """
    if not csv_path.exists():
        return False
    try:
        with open(csv_path) as fp:
            reader = csv.DictReader(fp)
            actual = sorted({float(r["max_time_ps"]) for r in reader if r.get("max_time_ps")})
    except (OSError, ValueError, KeyError):
        return False
    requested = sorted(requested_cutoffs)
    if len(actual) != len(requested):
        return False
    return all(abs(a - b) < 1e-3 for a, b in zip(actual, requested))

# Single-leg worker

def _run_one_leg(
    leg: Path,
    bar_deltae: Path,
    python_exe: str,
    time_cutoffs: str,
    default_temp: float,
    dry_run: bool,
) -> tuple[Path, str, str]:
    """Run bar_deltae.py for one leg. Returns (leg, status, message)."""
    nsim = detect_nsim(leg)
    if nsim is None:
        return leg, "error", "could not detect nsim (no para_conf.zsh, no reps.tar.gz, no prodrun/rep*/deltae.xvg)"
    temp = detect_temp(leg, default_temp)
    if _has_reps_tar(leg):
        # Prefer tar (single-file read).
        cmd = [
            python_exe, str(bar_deltae),
            "--xvgs", "reps/deltae_rep%sim.xvg",
            "--tar-file", str(leg / "reps.tar.gz"),
            "--nsim", str(nsim),
            "--temp", str(temp),
            "--save-dir", str(leg),
            "--time-cutoffs", time_cutoffs,
        ]
    else:
        # No archive — use raw prodrun/rep*/deltae.xvg files.
        cmd = [
            python_exe, str(bar_deltae),
            "--xvgs", str(leg / "prodrun" / "rep%sim" / "deltae.xvg"),
            "--nsim", str(nsim),
            "--temp", str(temp),
            "--save-dir", str(leg),
            "--time-cutoffs", time_cutoffs,
        ]
    if dry_run:
        return leg, "dry_run", " ".join(cmd)

    # Throttle BLAS/OpenMP to a single thread per subprocess. bar_deltae.py only
    # uses numpy/pymbar at modest scale; the default thread fan-out (~48 per
    # process on this node) multiplied by --jobs blew through ulimit -u and
    # triggered fork() EAGAIN failures in the first attempt.
    env = dict(os.environ)
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    env.setdefault("BLIS_NUM_THREADS", "1")

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            env=env,
        )
    except OSError as e:
        return leg, "error", f"subprocess failed: {e}"

    if result.returncode != 0:
        tail = "\n".join(result.stdout.splitlines()[-20:]) if result.stdout else ""
        return leg, "error", f"bar_deltae.py exit={result.returncode}; tail:\n{tail}"

    csv_path = leg / TIME_SERIES_CSV
    if not csv_path.exists():
        return leg, "error", "bar_deltae.py exit=0 but bar_time_series.csv missing"
    return leg, "ok", f"wrote {csv_path}"

# CLI

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Backfill bar_time_series.csv for FEP legs via bar_deltae.py --time-cutoffs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--base-dirs", nargs="+", default=None,
                   help="Root directories to scan recursively for reps.tar.gz. "
                        "Mutually exclusive with --leg-list.")
    p.add_argument("--leg-list", default=None,
                   help="Path to a file listing leg directories (one per line). "
                        "Use this when discovery via find is too slow or when "
                        "you want to restrict to a curated subset.")
    p.add_argument("--time-cutoffs", default=DEFAULT_TIME_CUTOFFS,
                   help=f"Comma-separated cutoffs in ps (default: {DEFAULT_TIME_CUTOFFS}).")
    p.add_argument("--temp", type=float, default=DEFAULT_TEMP,
                   help=f"Fallback temperature in K when ref-t is not detectable "
                        f"(default: {DEFAULT_TEMP}).")
    p.add_argument("--jobs", "-j", type=int, default=1,
                   help="Number of legs to process in parallel (process pool).")
    p.add_argument("--force", action="store_true",
                   help="Re-run bar_deltae.py even if bar_time_series.csv already has the requested cutoff set.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the commands that would be run; do not execute.")
    p.add_argument("--bar-deltae", default=str(DEFAULT_BAR_DELTAE),
                   help=f"Path to bar_deltae.py (default: {DEFAULT_BAR_DELTAE}).")
    p.add_argument("--python", default=_DEFAULT_FEPSUITE_PYTHON,
                   help="Python interpreter used to invoke bar_deltae.py. "
                        "Defaults to the fepsuite venv "
                        f"({_DEFAULT_FEPSUITE_PYTHON}); the mutmap venv ships "
                        "pymbar 4.x and is incompatible with bar_deltae.py's "
                        "pymbar 3.x API calls. Falls back to sys.executable "
                        "if the default path does not exist.")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args(argv)

def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    bar_deltae = Path(args.bar_deltae)
    if not bar_deltae.is_file():
        logger.error("bar_deltae.py not found at %s", bar_deltae)
        return 2

    python_exe = args.python
    if not Path(python_exe).is_file():
        logger.warning("Default fepsuite python not found at %s; falling back "
                       "to sys.executable (%s). bar_deltae.py may fail if "
                       "this env has pymbar >= 4.", python_exe, sys.executable)
        python_exe = sys.executable

    cutoffs = sorted({float(x) for x in args.time_cutoffs.split(",") if x.strip()})
    if not cutoffs:
        logger.error("Empty --time-cutoffs")
        return 2

    if (args.base_dirs is None) == (args.leg_list is None):
        logger.error("Exactly one of --base-dirs or --leg-list is required")
        return 2

    if args.leg_list:
        legs = load_leg_list(Path(args.leg_list))
        logger.info("Loaded %d leg(s) from %s", len(legs), args.leg_list)
    else:
        base_dirs = [Path(d).resolve() for d in args.base_dirs]
        logger.info("Scanning %d base dir(s) for reps.tar.gz ...", len(base_dirs))
        legs = find_legs(base_dirs)
        logger.info("Discovered %d candidate leg(s)", len(legs))

    pending: list[Path] = []
    skipped = 0
    for leg in legs:
        csv_path = leg / TIME_SERIES_CSV
        if not args.force and existing_csv_matches(csv_path, cutoffs):
            skipped += 1
            logger.debug("Skip (CSV already covers cutoffs): %s", leg)
            continue
        pending.append(leg)
    logger.info("%d leg(s) need (re)generation, %d already covered", len(pending), skipped)

    if not pending:
        return 0

    cutoffs_str = ",".join(str(c) for c in cutoffs)
    results: list[tuple[Path, str, str]] = []
    if args.jobs <= 1 or args.dry_run:
        for leg in pending:
            results.append(_run_one_leg(leg, bar_deltae, python_exe, cutoffs_str, args.temp, args.dry_run))
            logger.info("[%s] %s", results[-1][1], leg)
            if results[-1][1] == "error":
                logger.error("    %s", results[-1][2])
            elif results[-1][1] == "dry_run":
                logger.info("    %s", results[-1][2])
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = {
                pool.submit(_run_one_leg, leg, bar_deltae, python_exe, cutoffs_str, args.temp, False): leg
                for leg in pending
            }
            for fut in as_completed(futures):
                res = fut.result()
                results.append(res)
                logger.info("[%s] %s", res[1], res[0])
                if res[1] == "error":
                    logger.error("    %s", res[2])

    n_ok = sum(1 for _, s, _ in results if s == "ok")
    n_err = sum(1 for _, s, _ in results if s == "error")
    n_dry = sum(1 for _, s, _ in results if s == "dry_run")
    logger.info("Summary: ok=%d error=%d dry_run=%d skipped=%d total=%d",
                n_ok, n_err, n_dry, skipped, len(legs))
    return 0 if n_err == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
