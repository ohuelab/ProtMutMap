import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
from typing import List, Dict, Any, Tuple, Optional
from .tools.utils import is_nan_or_none

def generate_digraph_from_df(node_df: pd.DataFrame, edge_df: pd.DataFrame, node_keys: str = "mutations", edge_keys: List[str] = ["from_mutation", "to_mutation"]):
    graph = nx.DiGraph()
    for i, d in node_df.iterrows():
        node = d[node_keys]
        graph.add_node(node, **d.to_dict())

    for i, d in edge_df.iterrows():
        u = d[edge_keys[0]]
        v = d[edge_keys[1]]
        if u not in graph.nodes() or v not in graph.nodes():
            continue
        graph.add_edge(u, v, **d.to_dict())

    return graph

def generate_dataframe_from_digraph(graph: nx.DiGraph, node_key: str = "mutations", edge_keys: List[str] = ["from_mutation", "to_mutation"]):
    nodes_list = []
    edges_list = []
    for node, data in graph.nodes(data=True):
        data = {"node_key": node, **data}
        nodes_list.append(data)
    for u, v, data in graph.edges(data=True):
        data = {edge_keys[0]: u, edge_keys[1]: v, **data}
        edges_list.append(data)
    return pd.DataFrame(nodes_list), pd.DataFrame(edges_list)

def topological_layout(graph: nx.DiGraph):
    """Create a left-to-right flowing layout based on topological sorting."""
    # Perform topological sort
    try:
        topo_order = list(nx.topological_sort(graph))
    except (nx.NetworkXError, nx.NetworkXUnfeasible):
        raise ValueError("Graph contains cycles. Cannot perform topological sort.")

    # Calculate the level (depth) of each node
    levels = {}
    for node in topo_order:
        if not list(graph.predecessors(node)):  # If there are no predecessor nodes
            levels[node] = 0
        else:
            # Maximum level of predecessor nodes + 1
            levels[node] = max(levels[pred] for pred in graph.predecessors(node)) + 1

    # Group nodes by level
    from collections import defaultdict
    level_nodes = defaultdict(list)
    for node, level in levels.items():
        level_nodes[level].append(node)

    # Calculate positions
    pos = {}
    max_level = max(levels.values()) if levels else 0

    for level, nodes in level_nodes.items():
        x = (level / max_level) if max_level > 0 else 0

        # Arrange nodes of the same level vertically
        num_nodes = len(nodes)
        for i, node in enumerate(nodes):
            if num_nodes == 1:
                y = 1 / 2
            else:
                y = (i / (num_nodes - 1))
            pos[node] = (x, y)

    return pos

def safe_layout(graph: nx.DiGraph, fallback_layout: str = "spring") -> Dict[Any, Tuple[float, float]]:
    """
    Create a layout for a graph, using topological layout if possible,
    otherwise falling back to alternative layouts for graphs with cycles.

    Args:
        graph: NetworkX DiGraph
        fallback_layout: Layout to use when graph is not a DAG. Options: "spring", "kamada_kawai", "planar"

    Returns:
        Dictionary mapping nodes to (x, y) positions
    """
    # Try topological layout first
    try:
        return topological_layout(graph)
    except ValueError:
        # Graph contains cycles, use fallback layout
        import warnings
        warnings.warn(
            f"Graph contains cycles. Cannot use topological layout. "
            f"Using {fallback_layout} layout instead.",
            UserWarning
        )

        if fallback_layout == "spring":
            return nx.spring_layout(graph)
        elif fallback_layout == "kamada_kawai":
            try:
                return nx.kamada_kawai_layout(graph)
            except (nx.NetworkXError, ValueError):
                # Fallback to spring if kamada_kawai fails
                warnings.warn("kamada_kawai layout failed, using spring layout instead.", UserWarning)
                return nx.spring_layout(graph)
        elif fallback_layout == "planar":
            try:
                return nx.planar_layout(graph)
            except (nx.NetworkXError, ValueError):
                # Fallback to spring if planar fails
                warnings.warn("planar layout failed, using spring layout instead.", UserWarning)
                return nx.spring_layout(graph)
        else:
            raise ValueError(f"Unknown fallback_layout: {fallback_layout}. Use 'spring', 'kamada_kawai', or 'planar'.")

def draw_mutation_graph(
    graph: nx.DiGraph,
    pos: Optional[Dict[Any, Tuple[float, float]]] = None,
    highlight_nodes: List[Any] = ["WT"],
    ax: Optional[plt.Axes] = None,
    node_colors: Optional[Dict[Any, str]] = None,
    node_labels: Optional[Dict[Any, str]] = None,
    edge_labels: Optional[Dict[Tuple[Any, Any], str]] = None,
    mutation_key: str = "mutation_diff",
    ddG_key: str = "ddG",
    figsize: Tuple[float, float] = (10, 6),
    node_size: int = 500,
    edge_width: float = 1.0,
    edge_color: str = 'black',
):
    """
    Draw a mutation graph.

    Edge widths are automatically calculated as 0.1 * nrep for each edge.
    Edge colors are automatically determined based on nrep threshold:
    - Gray (default) for nrep < 32
    - Red for nrep >= 32 (large transformations like bulky/aromatic changes)

    Args:
        graph: nx.DiGraph with edge attributes including 'nrep' and mutation_key
        pos: Optional[Dict[Any, Tuple[float, float]]] - node positions
        highlight_nodes: List[Any] - nodes to label
        ax: Optional[plt.Axes] - matplotlib axes to draw on
        node_colors: Optional[Dict[Any, str]] - custom node colors
        node_labels: Optional[Dict[Any, str]] - custom node labels
        edge_labels: Optional[Dict[Tuple[Any, Any], str]] - custom edge labels
        mutation_key: str - key for mutation data in edge attributes (default: "mutation_diff")
        ddG_key: str - key for ddG data in edge attributes (default: "ddG")
        figsize: Tuple[float, float] - figure size if ax is None
        node_size: int - size of nodes
        edge_width: float - DEPRECATED: edge widths are now calculated from nrep (0.1 * nrep)
        edge_color: str - DEPRECATED: edge colors are now determined automatically (gray/red based on nrep >= 32)
    """
    # Set up default parameters
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    if pos is None:
        pos = nx.spring_layout(graph)

    # Set up node colors based on is_active and required attributes
    if node_colors is None:
        node_colors = []
        for node in graph.nodes():
            node_data = graph.nodes[node]
            if node_data.get("is_active"):
                node_colors.append('red')
            elif node_data.get("required"):
                node_colors.append('orange')
            else:
                node_colors.append('lightblue')

    if node_labels is None:
        # highlight_nodes defaults to ["WT"], which a reduced or degenerate
        # graph need not contain; labelling an absent node fails in networkx.
        node_labels = {node: node for node in highlight_nodes if node in graph}

    # Generate edge labels with mutation and ddG information
    if edge_labels is None:
        edge_labels = {}
        for source, target, data in graph.edges(data=True):
            edge_label = ""
            mutation_value = data.get(mutation_key)
            if not is_nan_or_none(mutation_value):
                edge_label += str(mutation_value)
            ddG_value = data.get(ddG_key)
            if not is_nan_or_none(ddG_value):
                if edge_label != "":
                    edge_label += "\n"
                edge_label += "{:.2f}".format(ddG_value)
            edge_labels[(source, target)] = edge_label

    # A caller may pass a layout computed from a different (e.g. pre-reduction)
    # graph, so place any node the layout does not cover instead of raising a
    # KeyError from networkx deep inside drawing.
    missing = [node for node in graph.nodes() if node not in pos]
    if missing:
        xs = [xy[0] for xy in pos.values()] or [0.0]
        x_extra = max(xs) + 1.0
        span = max(len(missing) - 1, 1)
        for index, node in enumerate(missing):
            pos[node] = (x_extra, 1.0 - 2.0 * index / span)

    # Calculate edge widths and colors based on nrep
    edge_widths = []
    edge_colors = []
    nrep_threshold = 32  # Threshold for red color (large transformations)
    for source, target, data in graph.edges(data=True):
        # Calculate width from nrep (0.1 * nrep)
        nrep = data.get("nrep", 1)
        if nrep is None or is_nan_or_none(nrep):
            nrep = 1
        nrep_float = float(nrep)
        edge_widths.append(0.1 * nrep_float)

        # Determine color: red for nrep >= threshold (large transformations), gray otherwise
        edge_colors.append('red' if nrep_float >= nrep_threshold else 'gray')

    # Draw graph components
    # nx.draw(graph, pos=pos, node_size=node_size, node_color=node_colors, ax=ax)
    nx.draw_networkx_nodes(graph, pos=pos, node_size=node_size, node_color=node_colors, ax=ax)
    nx.draw_networkx_edges(graph, pos=pos, edge_color=edge_colors, width=edge_widths, ax=ax)
    nx.draw_networkx_labels(graph, pos=pos, labels=node_labels, ax=ax)
    nx.draw_networkx_edge_labels(graph, pos=pos, edge_labels=edge_labels, ax=ax)

    return ax
