# FEP workflow

Use this workflow to turn a mutation network into binding ΔΔG observations
for the [node estimator](../README.md). Commands run from the repository root.

For the study's calculations, download the [FEP input archive](https://github.com/ohuelab/ProtMutMap/releases/download/fep-inputs/protmutmap_fep_inputs.zip).
Its `manifest.tsv` maps each directed edge and calculation leg to the corresponding
coordinates, topology, and referenced `.itp` files. The forward edges are listed in
[`edge_diagnostics.csv`](../results/source_data/fep/edge_diagnostics.csv).

## Requirements and structures

Install the preparation and BAR dependencies:

```bash
pip install -e ".[analysis,prep,bar]"
```

Install FEPsuite (including `fepgen` and its force fields), its HREX-patched
GROMACS, FASPR, and zsh. Prepare PDB files for the complex and both partners,
retaining consistent chain identifiers and residue numbering.

The example uses the supplied 1BJ1 structures and `mutations_1BJ1.txt`.
In `1BJ1_HL_VW`, `HL` identifies partner 1's chains and `VW` partner 2's
chains. Replace the identifier, structures and mutations together for your system.

## Prepare the calculations

Build the network:

```bash
protmutmap --mutations examples/mutations_1BJ1.txt \
  --experimental-dGs '{"WT": 0}' --output-dir out/1BJ1
```

Then supply the structures and absolute executable paths:

```bash
protmutmap-fepsuite-preparer \
  --link-file out/1BJ1/links.tsv \
  --target 1BJ1_HL_VW --base-target-dir work/1BJ1 \
  --complex-pdb examples/1BJ1.pdb \
  --partner1-pdb examples/1BJ1_HL.pdb \
  --partner2-pdb examples/1BJ1_VW.pdb \
  --fepsuite-dir /path/to/fepsuite \
  --faspr /path/to/FASPR --gmx /path/to/gmx
```

The preparer creates calculation directories under `work/1BJ1/complex/`,
`partner1/` and `partner2/`, and prints `calculation_files` and `failed_files`.
Check the failure list and error messages before running simulations.
A partner is calculated only when the transformation changes its chains.

## Run the simulations

Set the installation paths and copy the templates into a run directory:

```bash
export FEPSUITE_ROOT=/path/to/fepsuite
export GROMACS_DIR=/path/to/gromacs
mkdir -p run
cp fep_protocol/run.zsh fep_protocol/para_conf.zsh run/
cp -r fep_protocol/mdp run/
chmod +x run/run.zsh
```

In `run/para_conf.zsh`, set `PARA` (MPI processes per replica), `TPP` (threads
per process) and `SIMLENGTH` (ps) for your resources and simulation protocol.
Keep each calculation's generated `para_conf.zsh`; it overrides settings
such as the replica count.

On your compute resources, run each directory listed in `calculation_files`:

```bash
cd run
./run.zsh /absolute/path/to/calculation none 8
cd ..
```

`none` runs directly in the current environment; `8` is the final stage.
Completed stages are skipped using `done_step.txt`. Check completion,
`bar1.log` and simulation convergence before aggregating results.

## Collect edge values

This reads the completed calculations in the layout produced above and writes
binding ΔΔG as complex ΔG minus the affected partners' ΔG:

```python
from pathlib import Path
import pandas as pd
from protmutmap.gather_results import gather_results

links = pd.read_csv("out/1BJ1/links.tsv", sep="\t")
edges = gather_results(
    links, Path("work/1BJ1"),
    {"partner1": list("HL"), "partner2": list("VW")},
)
missing = edges["calc_ddG"].isna()
if missing.any():
    raise RuntimeError(f"Incomplete FEP results for {missing.sum()} edges")
edges[["from_mutation", "to_mutation", "calc_ddG"]].to_csv(
    "edges.csv", index=False,
)
```

Pass `edges.csv` to the estimator in the README. For time-specific estimates,
`gather_results(..., time_ps=4000, allow_bar1_fallback=False)` reads the matching
successful rows from each calculation's `bar_time_series.csv`; those series
must first be generated for the requested cutoff.

For reverse observations, add separate rows with the actual starting and
ending variants and corresponding signed ΔΔG. The Huber estimator fits both
observations separately. `fit.edges` provides residuals and weights for
inspecting the fitted network.
