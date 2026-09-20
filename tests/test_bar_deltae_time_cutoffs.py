"""Check BAR time windows, unit conversion and invalid cutoff handling."""

from __future__ import annotations

import importlib.util
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_BAR_DELTAE_PATH = _REPO_ROOT / "protmutmap" / "tools" / "bar_deltae.py"

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("pymbar") is None,
    reason="bar_deltae.py requires pymbar 3.x (optional extra 'bar')",
)


@pytest.fixture(scope="module")
def bar_deltae_module():
    spec = importlib.util.spec_from_file_location("bar_deltae", _BAR_DELTAE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- #
# argparse mutual exclusion
# --------------------------------------------------------------------------- #

def test_mutual_exclusion_time_cutoffs_and_max_time(tmp_path: Path):
    """--time-cutoffs and --max-time at the same time must exit non-zero."""
    result = subprocess.run(
        [
            sys.executable, str(_BAR_DELTAE_PATH),
            "--xvgs", "/nonexistent/deltae_rep%sim.xvg",
            "--nsim", "2",
            "--temp", "300",
            "--save-dir", str(tmp_path),
            "--time-cutoffs", "1000",
            "--max-time", "2000",
        ],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    assert result.returncode != 0
    combined = (result.stdout + result.stderr).lower()
    assert "--time-cutoffs and --max-time are mutually exclusive" in combined


# --------------------------------------------------------------------------- #
# compute_final_estimate — direct unit tests
# --------------------------------------------------------------------------- #

# Isolate time-window selection and unit conversion from the estimator.
_FAKE_BAR_RESULT_KBT = 0.7  # in units of beta·dG; arbitrary but stable


def _fake_bar(emat, time_all, nsim, btime, etime, show_intermediate):
    # Mirror the existing bar() guard: empty windows must error so callers
    # that mark status='error:ValueError' get reached.
    SimEval = type(next(iter(emat.keys())))
    basestate = SimEval(sim=0, eval=0)
    base_times = time_all[basestate]
    mask = np.logical_and(base_times > btime, base_times <= etime)
    if int(np.sum(mask)) == 0:
        raise AssertionError("fake bar() got empty window")
    return _FAKE_BAR_RESULT_KBT


def _synthetic_inputs(bd, nframes: int = 200, dt_ps: float = 10.0):
    SimEval = bd.SimEval
    times = np.arange(1, nframes + 1, dtype=float) * dt_ps
    # Values are irrelevant once bar() is monkeypatched.
    arr = np.zeros(nframes, dtype=float)
    energies = {
        SimEval(sim=0, eval=0): arr.copy(),
        SimEval(sim=0, eval=1): arr.copy(),
        SimEval(sim=1, eval=0): arr.copy(),
        SimEval(sim=1, eval=1): arr.copy(),
    }
    time_all = {k: times.copy() for k in energies}
    return energies, time_all, times


def test_compute_final_estimate_recovers_constant_energy_offset(bar_deltae_module):
    """A constant reduced-energy offset has an exact free-energy difference."""
    bd = bar_deltae_module
    energies, time_all, _ = _synthetic_inputs(bd, nframes=200, dt_ps=10.0)
    delta_kbt = 0.7
    energies[bd.SimEval(sim=0, eval=1)][:] = delta_kbt
    energies[bd.SimEval(sim=1, eval=0)][:] = -delta_kbt
    result = bd.compute_final_estimate(
        energies, time_all, nsim=2, split=10, temp=300.0,
        equilibration_time=None, tmax_cutoff=2000.0,
    )
    for key in ("dG_kcal", "dG_err_kcal", "t_start_ps", "t_end_ps",
                "n_split_chunks", "min_chunk_frames", "per_chunk"):
        assert key in result, f"missing key {key}"
    assert result["n_split_chunks"] == 10
    assert result["t_end_ps"] == pytest.approx(2000.0)
    # tstart defaults to (tmin + tmax)/2 = (10 + 2000)/2 = 1005
    assert result["t_start_ps"] == pytest.approx(1005.0)
    assert result["min_chunk_frames"] > 0
    assert len(result["per_chunk"]) == 10
    assert all(math.isfinite(v) for (_, _, v) in result["per_chunk"])
    expected_kcal = delta_kbt * bd.gasconstant * 300.0 / 4.184
    assert result["dG_kcal"] == pytest.approx(expected_kcal)
    assert result["dG_err_kcal"] == pytest.approx(0.0)


def test_compute_final_estimate_respects_equilibration_time(bar_deltae_module, monkeypatch):
    bd = bar_deltae_module
    monkeypatch.setattr(bd, "bar", _fake_bar)

    energies, time_all, _ = _synthetic_inputs(bd, nframes=200, dt_ps=10.0)
    r = bd.compute_final_estimate(
        energies, time_all, nsim=2, split=10, temp=300.0,
        equilibration_time=500.0, tmax_cutoff=2000.0,
    )
    # equilibration_time should win over (tmin+tmax)/2.
    assert r["t_start_ps"] == pytest.approx(500.0)
    assert r["t_end_ps"] == pytest.approx(2000.0)


def test_compute_final_estimate_raises_when_eq_time_exceeds_cutoff(bar_deltae_module):
    bd = bar_deltae_module
    SimEval = bd.SimEval
    nframes = 100
    times = np.arange(1, nframes + 1, dtype=float) * 10.0  # up to 1000 ps
    energies = {
        SimEval(sim=0, eval=0): np.zeros(nframes),
        SimEval(sim=0, eval=1): np.zeros(nframes),
        SimEval(sim=1, eval=0): np.zeros(nframes),
        SimEval(sim=1, eval=1): np.zeros(nframes),
    }
    time_all = {k: times.copy() for k in energies}

    with pytest.raises(ValueError):
        bd.compute_final_estimate(
            energies, time_all, nsim=2, split=10, temp=300.0,
            equilibration_time=1500.0, tmax_cutoff=1000.0,
        )


def test_compute_final_estimate_raises_on_zero_frame_chunk(bar_deltae_module):
    """When the cutoff is too small to fit `split` chunks with >=1 frame each,
    compute_final_estimate must raise — this is the mechanism the caller uses
    to mark the row status='error:ValueError' or 'too_short'.
    """
    bd = bar_deltae_module
    SimEval = bd.SimEval
    # Only 4 frames after equilibration → split=10 chunks impossible.
    nframes = 8
    times = np.arange(1, nframes + 1, dtype=float) * 10.0  # 10..80 ps
    beta = 1.0 / (bd.gasconstant * 300.0)
    rng = np.random.default_rng(0)
    energies = {
        SimEval(sim=0, eval=0): rng.normal(0.0, 0.05, nframes) * beta,
        SimEval(sim=0, eval=1): rng.normal(0.2, 0.05, nframes) * beta,
        SimEval(sim=1, eval=0): rng.normal(-0.2, 0.05, nframes) * beta,
        SimEval(sim=1, eval=1): rng.normal(0.0, 0.05, nframes) * beta,
    }
    time_all = {k: times.copy() for k in energies}

    with pytest.raises(ValueError):
        bd.compute_final_estimate(
            energies, time_all, nsim=2, split=10, temp=300.0,
            equilibration_time=None, tmax_cutoff=80.0,
        )
