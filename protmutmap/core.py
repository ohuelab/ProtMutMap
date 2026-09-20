from typing import Union, List, Dict, Optional, Set, Tuple
import networkx as nx
from copy import deepcopy
from .tools.mutations import MutationList
from .tools.lambda_calculator import calculate_nrep
from .tools.crystal_picker import pick_nearest_crystal

class ProtMutMap:
    def __init__(
        self,
        mutations_list: List[Union[MutationList, str]],
        experimental_dGs: Optional[Dict[str, float]] = None,
        max_mutation_num: int = 1,
        penalize_fewer_mutations: bool = False,
        is_cyclic: bool = True,
        max_cycle_size: int = 4,
        allow_bidirectional_cycles: bool = False,
        crystal_pdbs: Optional[Dict[str, Dict[str, str]]] = None,
    ):
        """
        Initialize ProtMutMap with mutation list and parameters.

        Args:
            mutations_list: List of MutationList objects or strings (required mutations)
            experimental_dGs: Dict mapping mutation strings to experimental dG values
            max_mutation_num: Maximum allowed mutations per edge (default: 1)
            penalize_fewer_mutations: Edge removal ordering flag (default: False)
                - False (default): Prefer removing complex multi-mutation edges (difficult or extreme)
                  when nrep_cost is equal. This prioritizes splitting complex mutations.
                - True: Always prefer removing single-mutation edges when nrep_cost is equal.
                  This prioritizes keeping multi-mutation edges.
            is_cyclic: Enable cycle constraint (default: True)
            max_cycle_size: Maximum cycle size for constraint (default: 4)
            allow_bidirectional_cycles: If True, allow bidirectional edges (size 2 cycles)
                to be counted as valid cycles (default: False)
            crystal_pdbs: Optional dict ``{mutation_str: {leg: pdb_path}}``. Nodes
                listed here are treated as having crystal structures available
                and are protected from removal. Edges attach a ``from_pdb_source``
                attribute indicating whether the source node uses a crystal start
                or is built by FASPR from a neighbouring crystal.
        """
        self.mutations_list = [
            MutationList.from_string(m) if isinstance(m, str) else m
            for m in mutations_list
        ]

        if experimental_dGs is None or len(experimental_dGs) == 0:
            self.experimental_dGs = {"WT": 0}
        else:
            self.experimental_dGs = experimental_dGs

        self.max_mutation_num = max_mutation_num
        self.penalize_fewer_mutations = penalize_fewer_mutations
        self.is_cyclic = is_cyclic
        self.max_cycle_size = max_cycle_size
        self.allow_bidirectional_cycles = allow_bidirectional_cycles
        self.crystal_pdbs: Dict[str, Dict[str, str]] = crystal_pdbs or {}

    def build_initial_map(self) -> nx.DiGraph:
        """
        Build initial mutation graph without removing edges or nodes.

        Returns:
            Initial mutation graph as nx.DiGraph
        """
        return self._build_initial_graph()

    def build_map(self) -> nx.DiGraph:
        """
        Build mutation graph and iteratively remove edges and nodes while satisfying constraints.

        Returns:
            Reduced mutation graph as nx.DiGraph
        """
        mutation_graph = self._build_initial_graph()

        mutation_graph = self._remove_edges(mutation_graph)

        mutation_graph = self._remove_nodes(mutation_graph)

        mutation_graph = self._remove_redundant_bidirectional_edges(mutation_graph)

        return mutation_graph

    def _build_initial_graph(self) -> nx.DiGraph:
        """Build the graph of mutation subsets."""
        mutation_graph = nx.DiGraph()

        for target_muts in self.mutations_list:
            for muts in target_muts.generate_combinations():
                mutations_str = muts.to_string() if muts.to_string() != "" else "WT"
                ddG = self.experimental_dGs.get(mutations_str)
                is_active = ddG is not None
                required = muts in self.mutations_list
                crystal_pdb = self.crystal_pdbs.get(mutations_str)

                mutation_graph.add_node(
                    mutations_str,
                    mutations=mutations_str,
                    ddG=ddG,
                    is_active=is_active,
                    required=required,
                    crystal_pdb=crystal_pdb,
                )

        # Include crystal reference nodes even when they are not target subsets.
        for crystal_mut, crystal_paths in self.crystal_pdbs.items():
            if crystal_mut in mutation_graph.nodes():
                continue
            ddG = self.experimental_dGs.get(crystal_mut)
            mutation_graph.add_node(
                crystal_mut,
                mutations=crystal_mut,
                ddG=ddG,
                is_active=ddG is not None,
                required=False,
                crystal_pdb=crystal_paths,
            )

        for source_node in mutation_graph.nodes():
            for target_node in mutation_graph.nodes():
                source_ddG = mutation_graph.nodes[source_node].get("ddG")
                target_ddG = mutation_graph.nodes[target_node].get("ddG")

                source_ml = MutationList.from_string(
                    source_node if source_node != "WT" else ""
                )
                target_ml = MutationList.from_string(
                    target_node if target_node != "WT" else ""
                )

                diff_muts = target_ml - source_ml
                num_diff_muts = len(diff_muts)

                # Only allow forward (addition-only) edges:
                # 1. Source positions must be a strict subset of target positions.
                # 2. For positions shared by both source and target, the mutation
                #    (after_res) must be identical — substituting one mutation for
                #    another at the same site (e.g. GD28T → GD28M) is not allowed.
                if num_diff_muts > 0:
                    source_muts_by_pos = {
                        (m.chain, m.resid, m.icode): m for m in source_ml
                    }
                    target_muts_by_pos = {
                        (m.chain, m.resid, m.icode): m for m in target_ml
                    }
                    source_positions = set(source_muts_by_pos)
                    target_positions = set(target_muts_by_pos)
                    if not source_positions.issubset(target_positions):
                        continue
                    if any(
                        source_muts_by_pos[pos].after_res != target_muts_by_pos[pos].after_res
                        for pos in source_positions
                    ):
                        continue

                if num_diff_muts > 0 and num_diff_muts <= self.max_mutation_num:
                    mutations = diff_muts.to_mutations()
                    nrep = calculate_nrep(mutations)

                    from .tools.lambda_calculator import assess_mutation_difficulty
                    is_difficult, has_charge_change, is_extreme = assess_mutation_difficulty(mutations)

                    if source_ddG is None or target_ddG is None:
                        ddG = None
                    else:
                        ddG = source_ddG - target_ddG

                    nrep_cost = nrep / num_diff_muts

                    from_pdb_source = self._resolve_pdb_source(source_node)

                    mutation_graph.add_edge(
                        source_node,
                        target_node,
                        mutation_diff=diff_muts.to_string(),
                        ddG=ddG,
                        num_diff_muts=num_diff_muts,
                        nrep=nrep,
                        nrep_cost=nrep_cost,
                        is_difficult=is_difficult,
                        has_charge_change=has_charge_change,
                        is_extreme=is_extreme,
                        from_pdb_source=from_pdb_source,
                    )

        return mutation_graph

    def _resolve_pdb_source(self, source_node: str) -> Optional[str]:
        """Decide where the FEP starting PDB comes from for ``source_node``.

        Returns ``"crystal"`` if the node itself has a registered crystal,
        otherwise ``"faspr:<crystal_mut_str>"`` naming the nearest crystal
        whose coordinates seed the FASPR build. Returns ``None`` if no
        crystals are configured at all.
        """
        if not self.crystal_pdbs:
            return None
        if source_node in self.crystal_pdbs:
            return "crystal"
        picked = pick_nearest_crystal(source_node, self.crystal_pdbs.keys())
        if picked is None:
            return None
        return f"faspr:{picked}"

    def _get_distance_from_active(self, node: str, graph: nx.DiGraph) -> float:
        """
        Calculate weighted distance from node to nearest active node (undirected).

        Args:
            node: Node name
            graph: Mutation graph

        Returns:
            Weighted distance (sum of num_diff_muts) or float('inf') if unreachable
        """
        active_nodes = [
            n for n in graph.nodes()
            if graph.nodes[n].get("is_active", False)
        ]

        if node in active_nodes:
            return 0.0

        if len(active_nodes) == 0:
            return float('inf')

        min_distance = float('inf')
        for active_node in active_nodes:
            distance = self._get_distance_between_nodes(node, active_node, graph)
            min_distance = min(min_distance, distance)

        return min_distance

    def _get_distance_between_nodes(
        self,
        node1: str,
        node2: str,
        graph: nx.DiGraph
    ) -> float:
        """
        Calculate weighted distance between two nodes (undirected).

        Args:
            node1: First node name
            node2: Second node name
            graph: Mutation graph

        Returns:
            Weighted distance (sum of num_diff_muts) or float('inf') if unreachable
        """
        if node1 == node2:
            return 0.0

        undirected_graph = graph.to_undirected()

        try:
            path_length = nx.shortest_path_length(
                undirected_graph,
                node1,
                node2,
                weight='num_diff_muts'
            )
            return path_length
        except nx.NetworkXNoPath:
            return float('inf')

    def _get_related_variant_group(
        self,
        required_node: str,
        graph: nx.DiGraph
    ) -> Optional[Set[str]]:
        """
        Find variant group related to a required node.

        Returns nodes on shortest path to nearest active node.
        If multiple paths exist, returns one of them.

        Args:
            required_node: Required node name
            graph: Mutation graph

        Returns:
            Set of node names in the variant group, or None if unreachable
        """
        active_nodes = [
            n for n in graph.nodes()
            if graph.nodes[n].get("is_active", False)
        ]

        if len(active_nodes) == 0:
            return None

        undirected_graph = graph.to_undirected()

        min_distance = float('inf')
        best_path = None

        for active_node in active_nodes:
            try:
                path_length = nx.shortest_path_length(
                    undirected_graph,
                    required_node,
                    active_node,
                    weight='num_diff_muts'
                )

                if path_length < min_distance:
                    min_distance = path_length
                    best_path = nx.shortest_path(
                        undirected_graph,
                        required_node,
                        active_node,
                        weight='num_diff_muts'
                    )
            except nx.NetworkXNoPath:
                continue

        if best_path is None:
            return None

        return set(best_path)

    def _check_cycle_constraint(
        self,
        variant_group: Set[str],
        graph: nx.DiGraph
    ) -> bool:
        """
        Verify cycle constraint for a variant group using undirected graph.

        According to the specification, the constraint is that edges between variant_group nodes
        should be part of cycles. We check that for each edge between variant_group nodes,
        there exists at least one cycle of size >= 3 and <= max_cycle_size that contains
        that edge.

        The constraint doesn't require all variant_group nodes to be in a single cycle,
        but rather that edges between variant_group nodes are part of cycles.
        Cycle detection is done on an undirected graph (direction doesn't matter).

        Examples:
        - AA→AW→WW→WA→AA: size 4 cycle containing edges between variant_group {AA, AW, WW}

        Args:
            variant_group: Set of node names in the variant group
            graph: Mutation graph

        Returns:
            True if constraint is satisfied, False otherwise
        """
        if not self.is_cyclic:
            return True

        if len(variant_group) < 2:
            return True

        undirected_graph = graph.to_undirected()

        variant_list = list(variant_group)
        variant_edges = set()
        for i, node1 in enumerate(variant_list):
            for node2 in variant_list[i+1:]:
                if undirected_graph.has_edge(node1, node2):
                    variant_edges.add(tuple(sorted([node1, node2])))

        if len(variant_edges) == 0:
            return True

        try:
            edges_in_cycles = set()

            for edge in variant_edges:
                node1, node2 = edge
                edge_found_in_cycle = False

                if self.allow_bidirectional_cycles and self.max_cycle_size >= 2:
                    if graph.has_edge(node1, node2) and graph.has_edge(node2, node1):
                        edges_in_cycles.add(edge)
                        edge_found_in_cycle = True
                        continue

                if not edge_found_in_cycle:
                    # If removing the edge disconnects the graph, it's not in a cycle
                    # Otherwise, find a path from node2 to node1 that doesn't use this edge
                    temp_graph = undirected_graph.copy()
                    if temp_graph.has_edge(node1, node2):
                        temp_graph.remove_edge(node1, node2)
                        try:
                            path = nx.shortest_path(temp_graph, node2, node1)
                            # Cycle size = path length (the path closes the cycle with the removed edge)
                            cycle_size = len(path)
                            if 3 <= cycle_size <= self.max_cycle_size:
                                edges_in_cycles.add(edge)
                                edge_found_in_cycle = True
                        except nx.NetworkXNoPath:
                            pass

            return len(edges_in_cycles) == len(variant_edges)
        except (nx.NetworkXError, nx.NetworkXNotImplemented):
            return True

    def _check_constraints(self, graph: nx.DiGraph) -> bool:
        """
        Validate all constraints after edge/node removal.

        Args:
            graph: Mutation graph to check

        Returns:
            True if all constraints are satisfied, False otherwise
        """
        required_nodes = [
            n for n in graph.nodes()
            if graph.nodes[n].get("required", False)
        ]
        active_nodes = [
            n for n in graph.nodes()
            if graph.nodes[n].get("is_active", False)
        ]

        # Constraint 1: Required nodes cannot be deleted (checked before deletion)
        # This is handled in removal methods

        # Constraint 2: All required nodes must be connected to at least one active node (undirected)
        if len(active_nodes) == 0:
            if "WT" not in self.experimental_dGs:
                return False
            if "WT" not in graph.nodes():
                return False

        undirected_graph = graph.to_undirected()
        for required_node in required_nodes:
            reachable = False
            for active_node in active_nodes:
                if nx.has_path(undirected_graph, required_node, active_node):
                    reachable = True
                    break
            if not reachable:
                return False

        # Constraint 3: Each required node must be reachable from at least one active
        # within weighted distance (sum of num_diff_muts along path)
        # The distance should be <= len(muts_required_node - muts_active_node) for at least one active
        for required_node in required_nodes:
            required_muts = MutationList.from_string(
                required_node if required_node != "WT" else ""
            )

            constraint_satisfied = False
            for active_node in active_nodes:
                active_muts = MutationList.from_string(
                    active_node if active_node != "WT" else ""
                )
                diff_muts = required_muts - active_muts
                min_required_distance = len(diff_muts)

                distance = self._get_distance_between_nodes(
                    required_node, active_node, graph
                )

                if distance <= min_required_distance:
                    constraint_satisfied = True
                    break

            if not constraint_satisfied:
                return False

        # Constraint 4: If is_cyclic=True, each required node's related variant group
        # must be in a cycle of size <= max_cycle_size (undirected graph)
        if self.is_cyclic:
            for required_node in required_nodes:
                variant_group = self._get_related_variant_group(required_node, graph)
                if variant_group is None:
                    return False

                if not self._check_cycle_constraint(variant_group, graph):
                    return False

        # Constraint 5: Every non-active, non-WT node must have at least one incoming
        # edge so that its ddG can be computed via FEP (no orphaned source nodes).
        for node in graph.nodes():
            if node == "WT":
                continue
            if graph.nodes[node].get("is_active", False):
                continue
            if graph.in_degree(node) == 0:
                return False

        return True

    def _sort_edges_for_removal(self, graph: nx.DiGraph) -> List[Tuple[str, str]]:
        """
        Sort edges according to removal priority rules.

        Priority:
        1. nrep_cost (nrep/num_diff_muts) descending
        2. If equal:
           - penalize_fewer_mutations=False (default): Prefer removing complex multi-mutation edges
             (num_diff_muts > 1 and (is_difficult or has_charge_change or is_extreme))
           - penalize_fewer_mutations=True: Prefer removing single-mutation edges (num_diff_muts == 1)
        3. If still equal, num_diff_muts ascending if penalize_fewer_mutations=True, else descending
        4. If still equal, distance from edge source to active nodes (descending)

        Args:
            graph: Mutation graph

        Returns:
            List of (source, target) edge tuples sorted by removal priority
        """
        edges = list(graph.edges())

        def sort_key(edge):
            source, target = edge
            edge_data = graph[source][target]
            nrep_cost = edge_data.get('nrep_cost', 0)
            num_diff_muts = edge_data.get('num_diff_muts', 1)
            is_difficult = edge_data.get('is_difficult', False)
            has_charge_change = edge_data.get('has_charge_change', False)
            is_extreme = edge_data.get('is_extreme', False)
            distance = self._get_distance_from_active(source, graph)

            # Edges connecting two required nodes are natural target-to-target links;
            # make them harder to remove by sorting them last.
            connects_required = int(
                graph.nodes[source].get("required", False) and
                graph.nodes[target].get("required", False)
            )

            # Primary: connects_required ascending (0 = non-required edge, tried first)
            # Secondary: nrep_cost descending
            # Tertiary:
            #   - penalize_fewer_mutations=False: prefer removing complex multi-mutation edges
            #   - penalize_fewer_mutations=True: prefer removing single-mutation edges
            # Quaternary: distance descending

            if self.penalize_fewer_mutations:
                is_single_mutation = (num_diff_muts == 1)
                return (connects_required, -nrep_cost, -is_single_mutation, num_diff_muts, -distance)
            else:
                is_complex_multi = (num_diff_muts > 1 and (is_difficult or has_charge_change or is_extreme))
                if is_complex_multi:
                    priority = 2
                elif num_diff_muts == 1:
                    priority = 1
                else:  # num_diff_muts > 1 and not complex
                    priority = 0
                return (connects_required, -nrep_cost, -priority, -distance)

        return sorted(edges, key=sort_key)

    def _remove_edges(self, graph: nx.DiGraph) -> nx.DiGraph:
        """
        Remove edges while satisfying constraints.

        Args:
            graph: Initial mutation graph

        Returns:
            Graph with edges removed
        """
        edges_to_check = self._sort_edges_for_removal(graph)

        for source, target in edges_to_check:
            graph_copy = deepcopy(graph)

            graph.remove_edge(source, target)

            if self._check_constraints(graph):
                continue
            else:
                graph = graph_copy

        return graph

    def _sort_nodes_for_removal(self, graph: nx.DiGraph) -> List[str]:
        """
        Sort nodes for removal (required nodes and crystal-bearing nodes excluded).

        Crystal-bearing nodes (``crystal_pdb`` attribute is not None) are
        protected because they provide a starting structure for FEP that we
        explicitly want to use.

        Args:
            graph: Mutation graph

        Returns:
            List of node names sorted for removal
        """
        nodes = [
            n for n in graph.nodes()
            if not graph.nodes[n].get("required", False)
            and graph.nodes[n].get("crystal_pdb") is None
        ]
        return nodes

    def _remove_nodes(self, graph: nx.DiGraph) -> nx.DiGraph:
        """
        Remove nodes while satisfying constraints (multi-pass until convergence).

        A single pass can leave removable nodes behind when their removal was
        blocked by a sibling node that gets removed in a later iteration.
        Repeating until no further node can be removed ensures the graph is
        fully pruned.

        Args:
            graph: Graph after edge removal

        Returns:
            Graph with nodes removed
        """
        changed = True
        while changed:
            changed = False
            nodes_to_check = self._sort_nodes_for_removal(graph)

            for node in nodes_to_check:
                # Node may have been removed in an earlier iteration of this pass
                if node not in graph.nodes():
                    continue

                if graph.nodes[node].get("required", False):
                    continue
                if graph.nodes[node].get("crystal_pdb") is not None:
                    continue

                graph_copy = deepcopy(graph)

                graph.remove_node(node)

                if self._check_constraints(graph):
                    changed = True
                else:
                    graph = graph_copy

        return graph

    def _remove_redundant_bidirectional_edges(self, graph: nx.DiGraph) -> nx.DiGraph:
        """
        Remove redundant bidirectional edges, keeping the one with lower removal priority.

        If both (A, B) and (B, A) edges exist, remove the one that comes earlier
        in the removal order (higher removal priority). This ensures that the edge
        with lower removal priority (which would be removed later) is kept.

        Args:
            graph: Mutation graph after edge and node removal

        Returns:
            Graph with redundant bidirectional edges removed
        """
        bidirectional_pairs = []
        processed_edges = set()

        for edge in graph.edges():
            source, target = edge
            reverse_edge = (target, source)


            if edge in processed_edges or reverse_edge in processed_edges:
                continue

            if graph.has_edge(target, source):
                bidirectional_pairs.append((edge, reverse_edge))
                processed_edges.add(edge)
                processed_edges.add(reverse_edge)

        if len(bidirectional_pairs) == 0:
            return graph

        sorted_edges = self._sort_edges_for_removal(graph)
        # Create a mapping from edge to its index in sorted order (lower index = higher priority = removed earlier)
        edge_to_index = {edge: idx for idx, edge in enumerate(sorted_edges)}

        edges_to_remove = []
        for edge1, edge2 in bidirectional_pairs:
            idx1 = edge_to_index.get(edge1, float('inf'))
            idx2 = edge_to_index.get(edge2, float('inf'))

            if idx1 < idx2:
                edges_to_remove.append(edge1)
            elif idx2 < idx1:
                edges_to_remove.append(edge2)
            else:
                edges_to_remove.append(edge1)

        for edge_to_remove in edges_to_remove:
            source, target = edge_to_remove
            graph_copy = deepcopy(graph)

            graph.remove_edge(source, target)

            if self._check_constraints(graph):
                continue
            else:
                graph = graph_copy

        return graph
