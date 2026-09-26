# ProtMutMap

ProtMutMap builds mutation networks and estimates protein–protein binding
ΔΔG from FEP edge values using Huber regression.

The [FEP input archive](https://github.com/ohuelab/ProtMutMap/releases/download/fep-inputs/protmutmap_fep_inputs.zip)
contains the calculation inputs used in the study. Its `manifest.tsv` maps
each directed mutation edge and calculation leg to the input files.

## Install

Requires Python 3.11 or later. Run from the repository root:

```bash
pip install -e ".[analysis]"
```

## 1. Build a network

Write one target variant per line in `mutations.txt`, separating substitutions
with commas. For example, `HH101Y,YH103W,SH105T` specifies three substitutions
in chain H. `HH101Y` means His → Tyr at PDB residue 101 of chain H.

```bash
protmutmap --mutations examples/mutations.txt \
  --experimental-dGs '{"WT": 0}' --output-dir out/
```

Replace `examples/mutations.txt` with your mutation list. The command writes
`node.tsv` (variants), `links.tsv` (FEP transformations), `graph.png` and
`graph.pkl`. This example supplies `{"WT": 0}`; supplying
experimental values for other variants changes the network reduction.

## 2. Estimate ΔΔG

Prepare `edges.csv` with one observation per row:

```csv
from_mutation,to_mutation,calc_ddG
WT,AA1T,-1.4
WT,AA2W,-2.1
AA1T,"AA1T,AA2W",-1.9
AA2W,"AA1T,AA2W",-1.1
```

`calc_ddG` is the binding free-energy change from the starting variant to the
ending variant, in kcal/mol. Use consistent variant names across rows.

```python
import pandas as pd
from protmutmap.robust_graph import fit_node_potentials

fit = fit_node_potentials(
    pd.read_csv("edges.csv"), ref_node="WT", method="huber",
    huber_delta=1.5, err_col=None, default_sigma=1.0,
)
if not fit.converged:
    raise RuntimeError("The node fit did not converge")
fit.nodes[["node", "energy"]].to_csv("predictions.csv", index=False)
```

`predictions.csv` contains the variant name (`node`) and predicted binding
ΔΔG relative to the reference variant (`energy`, kcal/mol). `ref_node` selects
the reference and fixes its `energy` to zero. Estimates cover its connected
component, ignoring edge direction. This example uses WT as the reference,
so negative values indicate stronger binding than WT.

To obtain the edge values from structures, follow the
[FEP workflow](docs/fep.md). It covers preparation, simulation setup and
conversion of BAR results into `edges.csv`.

## Documentation

- [Example inputs](examples/README.md)
- [Benchmark data](results/README.md)
- Command options: `protmutmap --help`

## License

ProtMutMap is distributed under [GPL-3.0-or-later](LICENSE).
The FEP preparation and analysis code includes code derived from
[FEPsuite](https://github.com/shunsakuraba/fepsuite).
`protmutmap/wcc/` is based on
[Weighted_cc](https://github.com/zlisysu/Weighted_cc)
(Copyright © 2022 zlisysu) and is distributed under the
[MIT licence](LICENSES/MIT-Weighted_cc.txt).
