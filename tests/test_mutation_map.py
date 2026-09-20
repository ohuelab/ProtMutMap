"""Tests for the DataFrame <-> graph conversion in protmutmap/mutation_map.py.

These conversions are the input/output boundary of the gather_results
pipeline: the node and edge tables written per system are read back as a graph
and written out again. A dropped column or attribute here silently changes
every downstream table, so the round trip is pinned.
"""

import networkx as nx
import pandas as pd
import pytest

from protmutmap.mutation_map import (
    generate_dataframe_from_digraph,
    generate_digraph_from_df,
)


@pytest.fixture
def node_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"mutations": "WT", "ddG": 0.0, "is_active": True, "required": False},
            {"mutations": "AA1T", "ddG": -1.5, "is_active": False, "required": False},
            {"mutations": "AA1T,AA2W", "ddG": -3.2, "is_active": False, "required": True},
        ]
    )


@pytest.fixture
def edge_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "from_mutation": "WT",
                "to_mutation": "AA1T",
                "mutation_diff": "AA1T",
                "nrep": 32,
                "num_diff_muts": 1,
            },
            {
                "from_mutation": "AA1T",
                "to_mutation": "AA1T,AA2W",
                "mutation_diff": "AA2W",
                "nrep": 48,
                "num_diff_muts": 1,
            },
        ]
    )


def test_digraph_from_df_keeps_every_column_as_an_attribute(node_df, edge_df):
    graph = generate_digraph_from_df(node_df, edge_df)

    assert set(graph.nodes()) == {"WT", "AA1T", "AA1T,AA2W"}
    assert set(graph.edges()) == {("WT", "AA1T"), ("AA1T", "AA1T,AA2W")}
    assert graph.nodes["AA1T"]["ddG"] == pytest.approx(-1.5)
    assert graph.nodes["AA1T,AA2W"]["required"] is True
    assert graph.edges["AA1T", "AA1T,AA2W"]["mutation_diff"] == "AA2W"
    assert graph.edges["AA1T", "AA1T,AA2W"]["nrep"] == 48


def test_round_trip_preserves_node_and_edge_values(node_df, edge_df):
    graph = generate_digraph_from_df(node_df, edge_df)
    nodes_out, edges_out = generate_dataframe_from_digraph(graph)

    # The node identifier is re-emitted as node_key alongside the original column.
    assert list(nodes_out["node_key"]) == list(node_df["mutations"])
    for column in node_df.columns:
        assert list(nodes_out[column]) == list(node_df[column]), column

    for column in edge_df.columns:
        assert list(edges_out[column]) == list(edge_df[column]), column


def test_edges_referencing_an_absent_node_are_dropped(node_df, edge_df):
    """Edge tables can outlive node removal, so dangling edges must not
    resurrect the node as an attribute-less placeholder."""
    dangling = pd.concat(
        [
            edge_df,
            pd.DataFrame([{"from_mutation": "AA1T", "to_mutation": "GONE"}]),
        ],
        ignore_index=True,
    )

    graph = generate_digraph_from_df(node_df, dangling)

    assert "GONE" not in graph.nodes()
    assert graph.number_of_edges() == 2


def test_dataframe_from_digraph_honours_custom_edge_column_names():
    graph = nx.DiGraph()
    graph.add_node("WT", ddG=0.0)
    graph.add_node("A", ddG=1.0)
    graph.add_edge("WT", "A", nrep=32)

    _, edges_out = generate_dataframe_from_digraph(graph, edge_keys=["src", "dst"])

    assert list(edges_out.columns[:2]) == ["src", "dst"]
    assert edges_out.iloc[0]["src"] == "WT"
    assert edges_out.iloc[0]["dst"] == "A"
