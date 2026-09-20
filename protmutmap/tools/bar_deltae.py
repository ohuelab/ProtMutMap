#!/usr/bin/env python

import pymbar
import sys
import re
import argparse
import numpy
import os.path
import pickle
import collections
import tarfile
import io

SimEval = collections.namedtuple("SimEval", ["sim", "eval"])

# in kJ/mol/K (to fit GROMACSy output)
gasconstant = 0.008314472
# in kcal/mol/K
gasconstant_kcal = 0.0019872036


def parse_args():
    parser = argparse.ArgumentParser(description = 'Convergence tester')
    parser.add_argument('--xvgs', metavar=".xvg", type = str, required=True,
                        help = 'xvg file path, "%sim", "%eval", and "%part" will be replaced by appropriate numbers')
    #parser.add_argument('--pvs', metavar=".xvg", type = str, required=True,
    #                    help = 'pV value files. "%sim" and "%part" will be replaced by appropriate numbers')
    parser.add_argument('--nsim', metavar="N", type = int, required=True,
                        help = 'number of simulations')
    parser.add_argument('--minpart', metavar="N", type = int, default=None,
                        help = 'part number begin')
    parser.add_argument('--maxpart', metavar="N", type = int, default=None,
                        help = 'part number end')
    parser.add_argument('--temp', help="Temperature (K)", type = float, default=300.0)
    parser.add_argument('--save-dir', help="save result to this directory", type = str, default = os.getcwd())
    parser.add_argument('--subsample', help="subsample interval", type = int, default = 1)
    parser.add_argument('--split', help="Number of chunks", type = int, default = 10)
    parser.add_argument('--show-intermediate', help="Show intermediate cumsum", action="store_true")
    parser.add_argument('--equilibration-time', help="Equilibration time to discard (ps). If not specified, uses latter half of simulation.", type = float, default = None)
    parser.add_argument('--max-time', help="Maximum simulation time (ps). Only samples up to this time will be used.", type = float, default = None)
    parser.add_argument('--tar-file', help="Path to tar.gz file containing xvg files. If specified, --xvgs should be the path pattern within the tar file (e.g., 'reps/deltae_rep%%sim.xvg')", type = str, default = None)
    parser.add_argument('--time-cutoffs', type=str, default=None,
                        help="Comma-separated max-time values in ps. Parses the deltae data once, "
                             "then re-runs the final-estimate BAR (latter-half SPLIT) at each cutoff "
                             "and writes bar_time_series.csv to --save-dir. "
                             "Mutually exclusive with --max-time.")
    parser.add_argument('--time-series-csv', type=str, default='bar_time_series.csv',
                        help="Filename inside --save-dir for the multi-cutoff CSV.")
    parser.add_argument('--min-chunk-frames', type=int, default=5,
                        help="Minimum frames required per split chunk for a cutoff row to be marked "
                             "status='ok'. Cutoffs that would produce fewer frames per chunk are "
                             "marked status='too_short'.")

    opts = parser.parse_args()
    if opts.time_cutoffs is not None and opts.max_time is not None:
        sys.exit("ERROR: --time-cutoffs and --max-time are mutually exclusive. "
                 "--time-cutoffs already performs per-cutoff truncation internally.")
    return opts

floatpat = r'[+-]?(?:\d+(\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'

import re
from math import isclose

def parse_deltae(fs, subsample, max_time=None):
    # Pattern for normal timestamps (any valid timestamp format)
    TIME_PAT_SAFE = re.compile(r'\d+\.\d{6}(?=\s|$)')
    # Pattern for rollback detection (lines starting with 100.xxxxxx)
    ROLLBACK_PAT  = re.compile(r'100\.\d{6}(?=\s|$)')

    # Patterns for the two different starting formats:
    # rep0, rep{nrep-1}: "100.000000    0    {float}    1    {float}\n100."
    # others:            "100.000000    0    {float}    1    {float}    2    {float}\n100."
    # We'll use these to detect the first valid line and handle corruption
    START_PAT_2STATE = re.compile(r'100\.000000\s+0\s+' + floatpat + r'\s+1\s+' + floatpat + r'(?:\s|$)')
    START_PAT_3STATE = re.compile(r'100\.000000\s+0\s+' + floatpat + r'\s+1\s+' + floatpat + r'\s+2\s+' + floatpat + r'(?:\s|$)')

    all_data = []
    for f in fs:
        file_data = []
        tprev = -1.0
        # Support both file paths (str) and file objects
        if isinstance(f, str):
            fh = open(f)
            should_close = True
        else:
            fh = f
            should_close = False
        try:
            samplecount = 0
            for raw in fh:
                if not raw or raw[0] in ['#', '@']:
                    continue
                line = raw.rstrip("\n")

                def try_parse(s):
                    parts = s.split()
                    if len(parts) < 3:
                        return None
                    try:
                        tt = float(parts[0])
                    except Exception:
                        return None
                    rest = parts[1:]
                    if len(rest) % 2 != 0:
                        return None
                    pair = []
                    seen_states = set()
                    try:
                        for i in range(0, len(rest), 2):
                            evix = int(rest[i])
                            evpot = float(rest[i+1])
                            # Check for duplicate state indices (indicates corruption)
                            if evix in seen_states:
                                return None
                            seen_states.add(evix)
                            pair.append((evix, evpot))
                    except Exception:
                        return None
                    return tt, pair

                parsed = try_parse(line)

                if parsed is None:
                    # Try to find where valid data starts in corrupted lines
                    m = None

                    # First, check if this might be the first line (starting with 100.000000)
                    # and extract it using the specific patterns
                    match_2state = START_PAT_2STATE.search(line)
                    match_3state = START_PAT_3STATE.search(line)

                    if match_2state or match_3state:
                        # Use whichever pattern matched
                        match = match_3state if match_3state else match_2state
                        repaired = line[match.start():]
                        parsed = try_parse(repaired)
                        if parsed is not None:
                            tt, pair = parsed
                            # Check for rollback condition (timestamp going backwards to 100.*)
                            if tprev >= 0 and tt < tprev:
                                file_data = []
                                samplecount = 0
                                tprev = -1.0
                        else:
                            continue
                    else:
                        # Not a start pattern, try general rollback or timestamp detection
                        rb = list(ROLLBACK_PAT.finditer(line))
                        if rb:
                            m = rb[-1]
                        else:
                            safes = list(TIME_PAT_SAFE.finditer(line))
                            if safes:
                                m = safes[-1]

                        if not m:
                            continue

                        repaired = line[m.start():]
                        parsed = try_parse(repaired)
                        if parsed is None:
                            continue

                        tt, pair = parsed

                        # Check for rollback condition
                        if tprev >= 0 and tt < tprev and ROLLBACK_PAT.fullmatch(repaired.split()[0]):
                            file_data = []
                            samplecount = 0
                            tprev = -1.0
                else:
                    tt, pair = parsed

                if tprev >= tt:
                    # Rollback: checkpoint-restart injected an out-of-order frame.
                    # Discard accumulated data from the previous segment and accept
                    # this frame as the start of the restarted run.
                    file_data = []
                    samplecount = 0
                    tprev = -1.0

                # Filter by max_time if specified
                if max_time is not None and tt > max_time:
                    continue

                if (samplecount % subsample) == 0:
                    for (evix, evpot) in pair:
                        file_data.append((tt, evix, evpot))
                samplecount += 1
                tprev = tt
        finally:
            if should_close:
                fh.close()

        all_data.extend(file_data)
    if all_data:
        print("DEBUG: all_data", len(all_data), all_data[-1])
    else:
        print("DEBUG: all_data EMPTY", file=sys.stderr)
    return all_data

def bar(emat, time_all, nsim, btime, etime, show_intermediate):
    dgtot = 0.0
    for isim in range(nsim - 1):
        basestate = SimEval(sim=isim, eval=isim)
        mask_isim = numpy.logical_and(time_all[basestate] > btime, time_all[basestate] <= etime)
        nmasked = int(numpy.sum(mask_isim))
        assert nmasked > 0, f"isim={isim}: no frames in window [{btime}, {etime}]"
        assert len(emat[basestate]) == len(emat[SimEval(sim=isim, eval=isim + 1)])
        # isim+1 may have a slightly different frame count if a replica was
        # killed and restarted; create its own time mask and take the minimum.
        base_next = SimEval(sim=isim+1, eval=isim+1)
        mask_next = numpy.logical_and(time_all[base_next] > btime, time_all[base_next] <= etime)
        nmasked_next = int(numpy.sum(mask_next))
        assert nmasked_next > 0, f"isim+1={isim+1}: no frames in window [{btime}, {etime}]"
        assert len(emat[base_next]) == len(emat[SimEval(sim=isim+1, eval=isim)])
        nmasked_common = min(nmasked, nmasked_next)
        # uses u_kn representation
        # K: evaluation states
        # N: samples (N_K)
        u = numpy.empty((2, nmasked_common * 2))
        u[0, 0:nmasked_common] = emat[basestate][mask_isim][:nmasked_common]
        u[1, 0:nmasked_common] = emat[SimEval(sim=isim, eval=isim+1)][mask_isim][:nmasked_common]
        u[0, nmasked_common:nmasked_common*2] = emat[SimEval(sim=isim+1, eval=isim)][mask_next][:nmasked_common]
        u[1, nmasked_common:nmasked_common*2] = emat[base_next][mask_next][:nmasked_common]

        nk = numpy.array([nmasked_common, nmasked_common])


        # print("debug shape:", u.shape, nk.shape)
        mb = pymbar.MBAR(u, nk, verbose=False)
        try:
            rettuple = mb.getFreeEnergyDifferences(compute_uncertainty=False, warning_cutoff=1)
            Deltaf = rettuple[0]
        except AttributeError:
            # pymbar >= 4: API changed
            result = mb.compute_free_energy_differences(compute_uncertainty=False)
            Deltaf = result["Delta_f"]
        #print("DEBUG", type(Deltaf))
        #print("DEBUG shape", Deltaf.shape)
        dgtot += Deltaf[0, 1]  # F[isim + 1] - F[isim]
        if show_intermediate:
            print("INT", isim, isim+1, Deltaf[0,1], dgtot)
    return dgtot # F[opts.nsim - 1] - F[0]

def compute_final_estimate(energies, time_all, nsim, split, temp,
                           equilibration_time, tmax_cutoff):
    """Run the latter-half SPLIT BAR final estimate for an arbitrary tmax cutoff.

    Mirrors the inline logic in main() (the section labeled 'Performing
    time-split BAR ...' followed by 'Final estimate'), but parameterized so
    callers can re-run it for multiple cutoffs without re-parsing deltae.xvg.

    Does not print; returns a dict for callers to format. tstart selection
    follows the same precedence as the inline block: explicit
    equilibration_time wins (clamped to >= tmin), otherwise (tmin + tmax)/2.
    """
    kcal_of_kJ = 1. / 4.184
    tmin = float(time_all[SimEval(0, 0)][0])
    tmax = float(tmax_cutoff)

    if equilibration_time is not None:
        tstart = float(equilibration_time)
        if tstart < tmin:
            tstart = tmin
        if tstart >= tmax:
            raise ValueError(
                "equilibration_time (%g ps) must be less than tmax_cutoff (%g ps)"
                % (tstart, tmax)
            )
    else:
        tstart = (tmin + tmax) / 2.0

    twidth = (tmax - tstart) / split
    per_chunk = []
    min_frames = None
    basestate = SimEval(sim=0, eval=0)
    base_times = time_all[basestate]
    for i in range(split):
        t0 = tstart + twidth * i
        t1 = tstart + twidth * (i + 1)
        mask = numpy.logical_and(base_times > t0, base_times <= t1)
        chunk_frames = int(numpy.sum(mask))
        if min_frames is None or chunk_frames < min_frames:
            min_frames = chunk_frames
        if chunk_frames == 0:
            raise ValueError(
                "split chunk %d (%g-%g ps) has 0 frames" % (i, t0, t1)
            )
        barres = bar(energies, time_all, nsim, t0, t1, False)
        per_chunk.append((t0, t1, barres * gasconstant * temp * kcal_of_kJ))

    vals = [v for (_, _, v) in per_chunk]
    femean = float(numpy.mean(vals))
    festderr = 0.0
    if split > 2:
        festderr = float(numpy.std(vals, ddof=1) / numpy.sqrt(split - 1))

    return {
        "dG_kcal": femean,
        "dG_err_kcal": festderr,
        "t_start_ps": tstart,
        "t_end_ps": tmax,
        "n_split_chunks": int(split),
        "min_chunk_frames": int(min_frames if min_frames is not None else 0),
        "per_chunk": per_chunk,
    }


def main():
    opts = parse_args()

    betas = []
    energies = {}
    nsamples = {}
    time_all = {}
    beta = 1. / (gasconstant * opts.temp)

    # Open tar file if specified
    tar_handle = None
    if opts.tar_file:
        if not os.path.exists(opts.tar_file):
            raise FileNotFoundError(f"Tar file not found: {opts.tar_file}")
        tar_handle = tarfile.open(opts.tar_file, 'r:gz')
        print(f"Reading from tar file: {opts.tar_file}", file=sys.stderr)

    try:
        for isim in range(opts.nsim):
            files = []
            if opts.tar_file:
                # Read from tar file
                if opts.minpart is not None:
                    for part in range(opts.minpart, opts.maxpart + 1):
                        tar_path = opts.xvgs.replace("%sim", str(isim)).replace("%part", "%04d" % part)
                        try:
                            member = tar_handle.getmember(tar_path)
                            file_obj = tar_handle.extractfile(member)
                            if file_obj is not None:
                                # Wrap in TextIOWrapper for text mode
                                files.append(io.TextIOWrapper(file_obj, encoding='utf-8'))
                        except KeyError:
                            pass
                else:
                    tar_path = opts.xvgs.replace("%sim", str(isim))
                    try:
                        member = tar_handle.getmember(tar_path)
                        file_obj = tar_handle.extractfile(member)
                        if file_obj is not None:
                            # Wrap in TextIOWrapper for text mode
                            files.append(io.TextIOWrapper(file_obj, encoding='utf-8'))
                    except KeyError:
                        pass
            else:
                # Read from regular files
                if opts.minpart is not None:
                    for part in range(opts.minpart, opts.maxpart + 1):
                        f = opts.xvgs.replace("%sim", str(isim)).replace("%part", "%04d"%part)
                        if os.path.exists(f):
                            files.append(f)
                else:
                    f = opts.xvgs.replace("%sim", str(isim))
                    if os.path.exists(f):
                        files.append(f)

            if not files:
                print(f"Warning: No files found for simulation {isim}", file=sys.stderr)
                continue

            data = parse_deltae(files, opts.subsample, opts.max_time)

            # Close file objects if they were opened from tar
            if opts.tar_file:
                for f in files:
                    if hasattr(f, 'close'):
                        f.close()

            for (t, st, energy) in data:
                # Additional max_time check (redundant but safe)
                if opts.max_time is not None and t > opts.max_time:
                    continue
                se = SimEval(sim=isim, eval=st)
                if se not in energies:
                    energies[se] = []
                    time_all[se] = []
                if len(time_all[se]) > 0 and time_all[se][-1] == t:
                    # dup frame
                    continue
                energies[se].append(energy)
                time_all[se].append(t)

            nsamples[isim] = len(time_all[SimEval(sim=isim, eval=isim)])
            # Format file names for display
            if opts.tar_file:
                file_display = opts.xvgs.replace("%sim", str(isim))
                if opts.minpart is not None:
                    file_display = file_display.replace("%part", "%04d" % opts.minpart)
            else:
                file_display = ", ".join(files)
            print("Finished loading %s with %d points %d frames" % (file_display, len(data), len(time_all[se]))) # time_all[se] shall exist
            sys.stdout.flush()
    finally:
        if tar_handle:
            tar_handle.close()

    # Since I am rewriting energies[k], just for the safety I don't use "for k in energies"
    for k in list(energies.keys()):
        energies[k] = numpy.array(energies[k]) * beta
        time_all[k] = numpy.array(time_all[k])

    # If any replica started late (e.g., after a checkpoint restart), trim all
    # arrays to the latest common start time so frame counts stay consistent.
    base_ses = [SimEval(sim=i, eval=i) for i in range(opts.nsim) if SimEval(sim=i, eval=i) in time_all]
    effective_tmin = max(float(time_all[se][0]) for se in base_ses if len(time_all[se]) > 0)
    rep0_tmin = float(time_all[SimEval(0, 0)][0])
    if effective_tmin > rep0_tmin:
        for k in list(time_all.keys()):
            mask = time_all[k] >= effective_tmin
            time_all[k] = time_all[k][mask]
            energies[k] = energies[k][mask]

    # When --time-cutoffs is given, trim to max cutoff before sliding BAR so that
    # replicas with slightly different end times all share the same frame range.
    if opts.time_cutoffs is not None:
        tcs = [float(x) for x in opts.time_cutoffs.split(",")]
        tmax_cutoff = max(tcs)
        for k in list(time_all.keys()):
            mask = time_all[k] <= tmax_cutoff
            time_all[k] = time_all[k][mask]
            energies[k] = energies[k][mask]

    # Do simple bar with 2x2 matrix
    print("Performing sliding-window (time x to 2x) sectioned BAR [kcal/mol]")
    kcal_of_kJ = 1. / 4.184

    tmin = time_all[SimEval(0,0)][0]
    tmax = time_all[SimEval(0,0)][-1]
    results_sliding = []
    for i in range(opts.split):
        twidth = (tmax - tmin) / opts.split
        tspan = (i + 1) * twidth
        t0 = tmin + 0.5 * tspan
        t1 = tmin + tspan
        barres = bar(energies, time_all, opts.nsim, t0, t1, opts.show_intermediate)
        print("BAR SLIDE %f-%f:" % (t0, t1), barres * gasconstant * opts.temp * kcal_of_kJ)
        results_sliding.append((t0, t1, barres))

    pickle.dump(results_sliding, open("%s/results-sliding.pickle" % opts.save_dir, "wb"))

    # Determine start time for final analysis
    if opts.equilibration_time is not None:
        tstart = opts.equilibration_time
        if tstart < tmin:
            print(f"Warning: equilibration_time ({tstart} ps) is before simulation start ({tmin} ps). Using {tmin} ps.", file=sys.stderr)
            tstart = tmin
        elif tstart >= tmax:
            raise ValueError(f"equilibration_time ({tstart} ps) must be less than simulation end ({tmax} ps)")
        print(f"Performing time-split BAR (equipartition from {tstart:.1f} ps to {tmax:.1f} ps) [kcal/mol]")
    else:
        tstart = (tmin + tmax) / 2
        print(f"Performing time-split BAR (equipartition of the latter half, from {tstart:.1f} ps to {tmax:.1f} ps) [kcal/mol]")
    twidth = (tmax - tstart) / opts.split
    results_normalsplit = []
    for i in range(opts.split):
        t0 = tstart + twidth * i
        t1 = tstart + twidth * (i + 1)
        barres = bar(energies, time_all, opts.nsim, t0, t1, opts.show_intermediate)
        print("BAR SPLIT %f-%f:" % (t0, t1), barres * gasconstant * opts.temp * kcal_of_kJ)
        results_normalsplit.append((t0, t1, barres))

    pickle.dump(results_normalsplit, open("%s/results-normalsplit.pickle" % opts.save_dir, "wb"))

    print("Final estimate [kcal/mol]")
    vals = [x * gasconstant * opts.temp * kcal_of_kJ for (_, _, x) in results_normalsplit]
    femean = numpy.mean(vals)
    festderr = 0.
    if opts.split > 2:
        festderr = numpy.std(vals, ddof=1) / numpy.sqrt(opts.split - 1)
    print("BAR %.2f %.2f" % (femean, festderr))

    # Reuse parsed energies to estimate each requested time cutoff.
    if opts.time_cutoffs is not None:
        import csv as _csv

        cutoffs = sorted({float(x) for x in opts.time_cutoffs.split(',') if x.strip()})
        tmin_data = float(time_all[SimEval(0, 0)][0])
        tmax_data = float(time_all[SimEval(0, 0)][-1])
        rows = []
        for tc in cutoffs:
            base_row = {
                "max_time_ps": tc,
                "dG_kcal": float("nan"),
                "dG_err_kcal": float("nan"),
                "t_start_ps": float("nan"),
                "t_end_ps": float("nan"),
                "n_split_chunks": opts.split,
                "min_chunk_frames": 0,
            }
            if tc <= tmin_data:
                rows.append({**base_row, "status": "below_tmin"})
                continue
            if tc > tmax_data + 1e-6:
                rows.append({**base_row, "t_end_ps": tmax_data, "status": "exceeds_tmax"})
                continue
            try:
                r = compute_final_estimate(
                    energies, time_all, opts.nsim, opts.split, opts.temp,
                    opts.equilibration_time, tc,
                )
                status = ("too_short"
                          if r["min_chunk_frames"] < opts.min_chunk_frames
                          else "ok")
                rows.append({
                    "max_time_ps": tc,
                    "dG_kcal": r["dG_kcal"],
                    "dG_err_kcal": r["dG_err_kcal"],
                    "t_start_ps": r["t_start_ps"],
                    "t_end_ps": r["t_end_ps"],
                    "n_split_chunks": r["n_split_chunks"],
                    "min_chunk_frames": r["min_chunk_frames"],
                    "status": status,
                })
            except Exception as exc:
                rows.append({**base_row, "status": "error:%s" % type(exc).__name__})

        cols = ["max_time_ps", "dG_kcal", "dG_err_kcal",
                "t_start_ps", "t_end_ps",
                "n_split_chunks", "min_chunk_frames", "status"]
        csv_path = os.path.join(opts.save_dir, opts.time_series_csv)
        with open(csv_path, "w", newline="") as fp:
            w = _csv.DictWriter(fp, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in cols})
        print("Wrote " + csv_path)


if __name__ == "__main__":
    main()

