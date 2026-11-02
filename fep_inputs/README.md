# FEP inputs

Input files for the FEP calculations reported in the paper.

- [`manifest.tsv`](manifest.tsv): mutation edges, calculation legs, input
  directories, and replica counts. Directory paths are relative to this directory.
- [`inputs/`](inputs/): coordinates, topologies, and referenced `.itp` files
  for each calculation.
- [`fep_protocol/`](fep_protocol/): run script, shared parameters, and MDP files.

See the [FEP workflow](../docs/fep.md) for software requirements and calculation
procedures.
