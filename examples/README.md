# Example: build a mutation network

This example constructs a network connecting four double mutants through WT
and intermediate variants. Use it to identify transformations for FEP.

## Run

From the repository root:

```bash
pip install -e ".[analysis]"
protmutmap --mutations examples/mutations.txt \
  --experimental-dGs examples/dGs_network.json --output-dir out/
```

The result has 11 nodes and 14 edges. Open `out/graph.png` to inspect how
target and intermediate variants connect. These edges specify transformations
to calculate; their ΔΔG values are obtained later by FEP.

## Understand the inputs

`mutations.txt` specifies four variants:

```text
AA1T,AA2W
AA1T,AA2V
AA1P,AA2V
AA1S,AA2R
```

Each line is one variant carrying all comma-separated substitutions.
`AA1T` means original residue A (Ala), chain A, residue 1, new residue T (Thr).
The first line therefore combines A1T and A2W in chain A. Blank lines and
comments beginning with `#` are skipped.

`dGs_network.json` contains `{"WT": 0}`. This example treats only WT as
known, with a value of zero. Other values supplied through
`--experimental-dGs` mark those variants as known and change the network reduction.

`dGs.json` illustrates a JSON mapping from variant names to ΔΔG values in
kcal/mol. Its numbers are illustrative; the command above does not use it.

## Use the outputs

| File | Contents and use |
|---|---|
| `out/graph.png` | Network diagram: variants are nodes and transformations are edges |
| `out/node.tsv` | WT, target and intermediate variants |
| `out/links.tsv` | Transformations to calculate: `from_mutation` is the starting variant, `to_mutation` the ending variant |
| `out/graph.pkl` | Saved network for use in Python |

The TSV files are tab-separated. To inspect the transformations:

```python
import pandas as pd
links = pd.read_csv("out/links.tsv", sep="\t")
print(links[["from_mutation", "to_mutation"]])
```

## Use your own system

Write your target variants in the same format and pass the file to
`--mutations`. Match chain identifiers, residue numbers and original amino
acids to the PDB structures you will use for FEP.

The supplied `1BJ1.pdb` is a complex; `1BJ1_HL.pdb` and `1BJ1_VW.pdb`
contain its H/L and V/W binding partners. The illustrative chain-A mutations
above do not correspond to these structures. Use `mutations_1BJ1.txt` for the structure-matched example
(H101Y, Y103W and S105T in chain H).

Continue with the [FEP workflow](../docs/fep.md) to calculate edges from
structures, or the [README](../README.md) to fit existing edge ΔΔG values.
