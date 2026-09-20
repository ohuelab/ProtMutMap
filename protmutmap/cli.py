#!/usr/bin/env python3
"""Build mutation networks and export graphs and tables."""
import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional

import networkx as nx

from .core import ProtMutMap
from .tools.mutations import MutationList

try:
    from .mutation_map import (
        draw_mutation_graph,
        generate_dataframe_from_digraph,
        safe_layout,
        topological_layout
    )
    _HAS_GRAPH = True
except ImportError:
    _HAS_GRAPH = False
    print("Warning: Graph visualization requires matplotlib and networkx", file=sys.stderr)

def load_mutations_list(mutations_input: str) -> List[MutationList]:
    """
    Load mutations list from string or file.

    Args:
        mutations_input: Comma-separated mutations string (treated as single multi-point mutation)
                        or path to file (one mutation set per line, comma-separated within line)

    Returns:
        List of MutationList objects
    """
    mutations_path = Path(mutations_input)

    if mutations_path.exists() and mutations_path.is_file():
        # Read from file (one mutation set per line, comma-separated within line)
        mutations = []
        with open(mutations_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    mutations.append(MutationList.from_string(line))
    else:
        # Parse as single mutation set (comma-separated mutations are treated as one multi-point mutation)
        mutations = [MutationList.from_string(mutations_input)]

    return mutations

def load_experimental_dGs(dGs_input: Optional[str]) -> Optional[Dict[str, float]]:
    """
    Load experimental dG values from JSON string or file.

    Args:
        dGs_input: JSON string or path to JSON file

    Returns:
        Dictionary mapping mutation strings to dG values, or None
    """
    if dGs_input is None:
        return None

    dGs_path = Path(dGs_input)

    if dGs_path.exists() and dGs_path.is_file():
        with open(dGs_path, 'r') as f:
            return json.load(f)
    else:
        return json.loads(dGs_input)

def load_crystal_pdbs(tsv_path: Optional[str]) -> Optional[Dict[str, Dict[str, str]]]:
    """Load crystal-PDB mapping from a TSV file.

    The TSV must have a header row beginning with ``mutation_str`` and
    additional columns naming legs (e.g. ``complex_pdb``, ``partner1_pdb``).
    Each row maps a canonical mutation string to per-leg PDB paths. The
    ``_pdb`` suffix on column names is stripped to derive the leg name used
    internally (``complex``, ``partner1``, ...). Relative paths are resolved
    against the TSV file's directory.

    Returns
    -------
    ``{mutation_str: {leg: pdb_path_str}}`` or ``None`` if ``tsv_path`` is
    ``None``.
    """
    if tsv_path is None:
        return None
    path = Path(tsv_path)
    if not path.exists():
        raise FileNotFoundError(f"Crystal PDB TSV not found: {path}")
    base_dir = path.parent
    import csv
    crystals: Dict[str, Dict[str, str]] = {}
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames is None or "mutation_str" not in reader.fieldnames:
            raise ValueError(
                f"Crystal PDB TSV must have a 'mutation_str' header column "
                f"(got fields: {reader.fieldnames})"
            )
        leg_fields = [
            f for f in reader.fieldnames
            if f != "mutation_str" and f.endswith("_pdb")
        ]
        if not leg_fields:
            raise ValueError(
                "Crystal PDB TSV must have at least one *_pdb column "
                "(e.g. complex_pdb, partner1_pdb, partner2_pdb)"
            )
        for row in reader:
            mut = row["mutation_str"].strip()
            if not mut:
                continue
            legs: Dict[str, str] = {}
            for field in leg_fields:
                value = (row.get(field) or "").strip()
                if not value:
                    continue
                leg = field[:-len("_pdb")]
                pdb_path = Path(value)
                if not pdb_path.is_absolute():
                    pdb_path = (base_dir / pdb_path).resolve()
                legs[leg] = str(pdb_path)
            crystals[mut] = legs
    return crystals

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Build mutation graph and save outputs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with single multi-point mutation
  protmutmap --mutations "HH101Y,YH103W,SH105T" --output-dir ./output

  # With mutations from file (one mutation set per line)
  protmutmap --mutations mutations.txt --output-dir ./output

  # With experimental dG values
  protmutmap --mutations "HH101Y,YH103W,SH105T" --experimental-dGs '{"WT": 0, "HH101Y,YH103W,SH105T": -1.5}' --output-dir ./output

  # With experimental dG values from file
  protmutmap --mutations mutations.txt --experimental-dGs dGs.json --output-dir ./output
        """
    )

    parser.add_argument(
        '--mutations',
        required=True,
        help='Comma-separated mutations string (treated as single multi-point mutation) or path to file (one mutation set per line)'
    )

    parser.add_argument(
        '--experimental-dGs',
        default=None,
        help='JSON string or path to JSON file with experimental dG values (e.g., \'{"WT": 0, "AA1T": -1.5}\')'
    )

    parser.add_argument(
        '--output-dir',
        required=True,
        help='Output directory for graph.pkl, node.tsv, links.tsv, and graph.png'
    )

    parser.add_argument(
        '--max-mutation-num',
        type=int,
        default=1,
        help='Maximum allowed mutations per edge (default: 1)'
    )

    parser.add_argument(
        '--penalize-fewer-mutations',
        action='store_true',
        help='Prefer removing single-mutation edges when nrep_cost is equal. '
             'By default (False), complex multi-mutation edges (difficult or extreme) are removed first.'
    )

    parser.add_argument(
        '--is-cyclic',
        action='store_true',
        default=True,
        help='Enable cycle constraint (default: True)'
    )

    parser.add_argument(
        '--no-cyclic',
        dest='is_cyclic',
        action='store_false',
        help='Disable cycle constraint'
    )

    parser.add_argument(
        '--max-cycle-size',
        type=int,
        default=4,
        help='Maximum cycle size for constraint (default: 4)'
    )

    parser.add_argument(
        '--allow-bidirectional-cycles',
        action='store_true',
        default=False,
        help='Allow bidirectional edges (size 2 cycles) to be counted as valid cycles (default: False)'
    )

    parser.add_argument(
        '--layout',
        choices=['topological', 'spring', 'kamada_kawai', 'planar'],
        default='topological',
        help='Layout method for graph visualization (default: topological)'
    )

    parser.add_argument(
        '--crystal-pdbs',
        default=None,
        help=(
            'Optional path to a TSV mapping canonical mutation_str to per-leg '
            'crystal PDB paths. Columns: mutation_str, complex_pdb, '
            'partner1_pdb, partner2_pdb (extra *_pdb columns allowed). '
            'Nodes listed here are protected from removal and prep uses the '
            'given PDB directly. Other nodes seed FASPR from the nearest '
            'crystal (alphabetical tie-break).'
        )
    )

    args = parser.parse_args()

    try:
        mutations_list = load_mutations_list(args.mutations)
        if not mutations_list:
            print("Error: No mutations found", file=sys.stderr)
            sys.exit(1)
    except Exception as e:
        print(f"Error loading mutations: {e}", file=sys.stderr)
        sys.exit(1)

    experimental_dGs = None
    if args.experimental_dGs:
        try:
            experimental_dGs = load_experimental_dGs(args.experimental_dGs)
        except Exception as e:
            print(f"Error loading experimental dG values: {e}", file=sys.stderr)
            sys.exit(1)

    crystal_pdbs = None
    if args.crystal_pdbs:
        try:
            crystal_pdbs = load_crystal_pdbs(args.crystal_pdbs)
            print(
                f"Loaded {len(crystal_pdbs)} crystal node(s): "
                f"{', '.join(sorted(crystal_pdbs))}"
            )
        except Exception as e:
            print(f"Error loading crystal PDB TSV: {e}", file=sys.stderr)
            sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Building mutation graph with {len(mutations_list)} required mutations...")
    mutmap = ProtMutMap(
        mutations_list=mutations_list,
        experimental_dGs=experimental_dGs,
        max_mutation_num=args.max_mutation_num,
        penalize_fewer_mutations=args.penalize_fewer_mutations,
        is_cyclic=args.is_cyclic,
        max_cycle_size=args.max_cycle_size,
        allow_bidirectional_cycles=args.allow_bidirectional_cycles,
        crystal_pdbs=crystal_pdbs,
    )

    mutation_graph = mutmap.build_map()
    print(f"Graph built: {len(mutation_graph.nodes())} nodes, {len(mutation_graph.edges())} edges")

    graph_pkl_path = output_dir / "graph.pkl"
    with open(graph_pkl_path, 'wb') as f:
        pickle.dump(mutation_graph, f)
    print(f"Saved graph to {graph_pkl_path}")

    if _HAS_GRAPH:
        node_df, edge_df = generate_dataframe_from_digraph(
            mutation_graph,
            node_key="mutations",
            edge_keys=["from_mutation", "to_mutation"]
        )

        # Flatten the crystal_pdb dict into per-leg columns for readable TSV.
        if "crystal_pdb" in node_df.columns:
            leg_keys = set()
            for value in node_df["crystal_pdb"]:
                if isinstance(value, dict):
                    leg_keys.update(value.keys())
            for leg in sorted(leg_keys):
                col = f"crystal_pdb_{leg}"
                node_df[col] = node_df["crystal_pdb"].apply(
                    lambda v, _leg=leg: v.get(_leg) if isinstance(v, dict) else None
                )
            node_df = node_df.drop(columns=["crystal_pdb"])

        node_tsv_path = output_dir / "node.tsv"
        node_df.to_csv(node_tsv_path, sep='\t', index=False)
        print(f"Saved nodes to {node_tsv_path}")

        links_tsv_path = output_dir / "links.tsv"
        edge_df.to_csv(links_tsv_path, sep='\t', index=False)
        print(f"Saved links to {links_tsv_path}")

        graph_png_path = output_dir / "graph.png"

        if args.layout == 'topological':
            try:
                pos = topological_layout(mutation_graph)
            except (ValueError, nx.NetworkXUnfeasible, nx.NetworkXError):
                print("Warning: Graph contains cycles, using safe_layout instead", file=sys.stderr)
                pos = safe_layout(mutation_graph)
        else:
            pos = safe_layout(mutation_graph, fallback_layout=args.layout)

        ax = draw_mutation_graph(mutation_graph, pos=pos)
        ax.figure.savefig(graph_png_path, dpi=300, bbox_inches='tight')
        print(f"Saved graph visualization to {graph_png_path}")
    else:
        print("Warning: Graph visualization not available (matplotlib/networkx not installed)", file=sys.stderr)
        print("Skipping node.tsv, links.tsv, and graph.png generation", file=sys.stderr)

    print("Done!")

if __name__ == "__main__":
    main()
