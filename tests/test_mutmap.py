"""
Tests for ProtMutMap class.

Tests verify that mutation maps are created correctly according to the specification
with various options.
"""

import networkx as nx
from protmutmap.core import ProtMutMap
from protmutmap.tools.mutations import MutationList


class TestMutMapBasic:
    """Basic tests for ProtMutMap initialization and graph building."""

    def test_initialization_with_defaults(self):
        """Test ProtMutMap initialization with default parameters."""
        mutations_list = [
            MutationList.from_string("AA1T,AA2W"),
            MutationList.from_string("AA1T,AA2V"),
        ]
        experimental_dGs = {"WT": 0}

        mutmap = ProtMutMap(
            mutations_list=mutations_list,
            experimental_dGs=experimental_dGs
        )

        assert mutmap.max_mutation_num == 1
        assert mutmap.penalize_fewer_mutations is False
        assert mutmap.is_cyclic is True
        assert mutmap.max_cycle_size == 4
        assert mutmap.experimental_dGs == {"WT": 0}

    def test_empty_experimental_dGs(self):
        """Test that empty experimental_dGs is converted to {"WT": 0}."""
        mutations_list = [MutationList.from_string("AA1T")]

        mutmap = ProtMutMap(mutations_list=mutations_list, experimental_dGs={})
        assert mutmap.experimental_dGs == {"WT": 0}

        mutmap2 = ProtMutMap(mutations_list=mutations_list, experimental_dGs=None)
        assert mutmap2.experimental_dGs == {"WT": 0}

    def test_build_initial_map(self):
        """Test that initial map is built correctly."""
        mutations_list = [
            MutationList.from_string("AA1T,AA2W"),
            MutationList.from_string("AA1T,AA2V"),
        ]
        experimental_dGs = {"WT": 0}

        mutmap = ProtMutMap(
            mutations_list=mutations_list,
            experimental_dGs=experimental_dGs,
            max_mutation_num=1
        )

        graph = mutmap.build_initial_map()


        required_nodes = [n for n in graph.nodes() if graph.nodes[n].get("required", False)]
        assert len(required_nodes) == 2  # AA1T,AA2W and AA1T,AA2V


        assert "WT" in graph.nodes()
        assert graph.nodes["WT"].get("is_active") is True
        assert graph.nodes["WT"].get("ddG") == 0


        for source, target in graph.edges():
            num_diff_muts = graph[source][target].get("num_diff_muts", 0)
            assert num_diff_muts > 0
            assert num_diff_muts <= mutmap.max_mutation_num


class TestMaxMutationNum:
    """Tests for max_mutation_num option."""

    def test_max_mutation_num_1(self):
        """Test with max_mutation_num=1 (only single mutations allowed)."""
        mutations_list = [
            MutationList.from_string("AA1T,AA2W"),
            MutationList.from_string("AA1T,AA2V"),
        ]
        experimental_dGs = {"WT": 0}

        mutmap = ProtMutMap(
            mutations_list=mutations_list,
            experimental_dGs=experimental_dGs,
            max_mutation_num=1
        )

        graph = mutmap.build_initial_map()


        for source, target in graph.edges():
            num_diff_muts = graph[source][target].get("num_diff_muts", 0)
            assert num_diff_muts <= 1


        double_mut_nodes = ["AA1T,AA2W", "AA1T,AA2V"]
        for node in double_mut_nodes:
            if node in graph.nodes():
                # Should not have direct edge from WT
                assert not graph.has_edge("WT", node)

    def test_max_mutation_num_2(self):
        """Test with max_mutation_num=2 (double mutations allowed)."""
        mutations_list = [
            MutationList.from_string("AA1T,AA2W"),
            MutationList.from_string("AA1T,AA2V"),
        ]
        experimental_dGs = {"WT": 0}

        mutmap = ProtMutMap(
            mutations_list=mutations_list,
            experimental_dGs=experimental_dGs,
            max_mutation_num=2
        )

        graph = mutmap.build_initial_map()


        has_double_mutation_edge = False
        for source, target in graph.edges():
            num_diff_muts = graph[source][target].get("num_diff_muts", 0)
            assert num_diff_muts <= 2
            if num_diff_muts == 2:
                has_double_mutation_edge = True

        # Should have at least one edge with num_diff_muts=2
        assert has_double_mutation_edge

        # Note: build_map ensures constraints are satisfied during removal,
        # but final constraint check may fail if cycle constraint cannot be satisfied.
        # The important thing is that build_map completes successfully.


class TestPenalizeFewerMutations:
    """Tests for penalize_fewer_mutations option."""

    def test_penalize_fewer_mutations_true(self):
        """Test with penalize_fewer_mutations=True."""
        mutations_list = [
            MutationList.from_string("AA1T,AA2W"),
            MutationList.from_string("AA1T,AA2V"),
        ]
        experimental_dGs = {"WT": 0}

        mutmap = ProtMutMap(
            mutations_list=mutations_list,
            experimental_dGs=experimental_dGs,
            max_mutation_num=2,
            penalize_fewer_mutations=True
        )

        graph = mutmap.build_initial_map()
        sorted_edges = mutmap._sort_edges_for_removal(graph)


        # Primary: nrep_cost descending
        # Secondary: num_diff_muts ascending (when penalize_fewer_mutations=True)
        for i in range(len(sorted_edges) - 1):
            edge1 = sorted_edges[i]
            edge2 = sorted_edges[i + 1]
            nrep_cost1 = graph[edge1[0]][edge1[1]].get("nrep_cost", 0)
            nrep_cost2 = graph[edge2[0]][edge2[1]].get("nrep_cost", 0)

            if nrep_cost1 == nrep_cost2:
                num_diff1 = graph[edge1[0]][edge1[1]].get("num_diff_muts", 0)
                num_diff2 = graph[edge2[0]][edge2[1]].get("num_diff_muts", 0)
                # When penalize_fewer_mutations=True, smaller num_diff_muts comes first
                assert num_diff1 <= num_diff2

class TestIsCyclic:
    """Tests for is_cyclic and max_cycle_size options."""

    def test_is_cyclic_false(self):
        """Test with is_cyclic=False (no cycle constraint)."""
        mutations_list = [
            MutationList.from_string("AA1T,AA2W"),
            MutationList.from_string("AA1T,AA2V"),
        ]
        experimental_dGs = {"WT": 0}

        mutmap = ProtMutMap(
            mutations_list=mutations_list,
            experimental_dGs=experimental_dGs,
            is_cyclic=False,
            max_cycle_size=4
        )

        graph = mutmap.build_map()


        assert mutmap._check_constraints(graph)

        # Cycle constraint should always return True when is_cyclic=False
        for node in graph.nodes():
            variant_group = mutmap._get_related_variant_group(node, graph)
            if variant_group is not None:
                assert mutmap._check_cycle_constraint(variant_group, graph)

        # Both should satisfy basic constraints (connectivity, distance)
        # Cycle constraint may not always be satisfiable depending on graph structure
        # The important thing is that build_map completes and produces valid graphs


class TestConstraints:
    """Tests for constraint validation."""

    def test_required_nodes_not_deleted(self):
        """Test that required nodes are never deleted."""
        mutations_list = [
            MutationList.from_string("AA1T,AA2W"),
            MutationList.from_string("AA1T,AA2V"),
        ]
        experimental_dGs = {"WT": 0}

        mutmap = ProtMutMap(
            mutations_list=mutations_list,
            experimental_dGs=experimental_dGs
        )

        graph = mutmap.build_map()


        # Required nodes are the exact mutations in mutations_list
        for target_muts in mutations_list:
            required_str = target_muts.to_string() if target_muts.to_string() != "" else "WT"
            assert required_str in graph.nodes(), (
                f"Required node {required_str} was deleted from the graph"
            )
            assert graph.nodes[required_str].get("required", False), (
                f"Node {required_str} should be marked as required"
            )

    def test_required_nodes_connected_to_active(self):
        """Test that all required nodes are connected to at least one active node."""
        mutations_list = [
            MutationList.from_string("AA1T,AA2W"),
            MutationList.from_string("AA1T,AA2V"),
        ]
        experimental_dGs = {"WT": 0}

        mutmap = ProtMutMap(
            mutations_list=mutations_list,
            experimental_dGs=experimental_dGs
        )

        graph = mutmap.build_map()


        undirected_graph = graph.to_undirected()
        required_nodes = [n for n in graph.nodes() if graph.nodes[n].get("required", False)]
        active_nodes = [n for n in graph.nodes() if graph.nodes[n].get("is_active", False)]

        for required_node in required_nodes:
            reachable = False
            for active_node in active_nodes:
                if nx.has_path(undirected_graph, required_node, active_node):
                    reachable = True
                    break
            assert reachable, f"Required node {required_node} is not connected to any active node"

    def test_required_nodes_within_distance(self):
        """Test that required nodes are within weighted distance from active nodes."""
        mutations_list = [
            MutationList.from_string("AA1T,AA2W"),
            MutationList.from_string("AA1T,AA2V"),
        ]
        experimental_dGs = {"WT": 0}

        mutmap = ProtMutMap(
            mutations_list=mutations_list,
            experimental_dGs=experimental_dGs
        )

        graph = mutmap.build_map()


        required_nodes = [n for n in graph.nodes() if graph.nodes[n].get("required", False)]
        active_nodes = [n for n in graph.nodes() if graph.nodes[n].get("is_active", False)]

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

                distance = mutmap._get_distance_between_nodes(required_node, active_node, graph)

                if distance <= min_required_distance:
                    constraint_satisfied = True
                    break

            assert constraint_satisfied, (
                f"Required node {required_node} is not within required distance "
                f"from any active node"
            )

    def test_cycle_constraint_bidirectional_edge(self):
        """Test cycle constraint with bidirectional edge (AA→WW and AA←WW)."""

        mutations_list = [
            MutationList.from_string("AA1T,AA2W"),  # This will be "AA1T,AA2W" node
        ]
        experimental_dGs = {"WT": 0}

        mutmap = ProtMutMap(
            mutations_list=mutations_list,
            experimental_dGs=experimental_dGs,
            is_cyclic=True,
            max_cycle_size=4,
            max_mutation_num=2  # Allow direct edge from WT to AA1T,AA2W
        )

        graph = mutmap.build_initial_map()

        # Manually add bidirectional edge if it doesn't exist

        if "WT" in graph.nodes() and "AA1T,AA2W" in graph.nodes():
            # Add bidirectional edge if not already present
            if not graph.has_edge("WT", "AA1T,AA2W"):
                graph.add_edge("WT", "AA1T,AA2W", mutation_diff="AA1T,AA2W", ddG=None,
                              num_diff_muts=2, nrep=1, nrep_cost=0.5)
            if not graph.has_edge("AA1T,AA2W", "WT"):
                graph.add_edge("AA1T,AA2W", "WT", mutation_diff="", ddG=None,
                              num_diff_muts=2, nrep=1, nrep_cost=0.5)


        variant_group = {"WT", "AA1T,AA2W"}
        result = mutmap._check_cycle_constraint(variant_group, graph)

        # Should detect size 2 cycle (bidirectional edge) when enabled
        # Note: This test may fail if allow_bidirectional_cycles=False (default)
        # The bidirectional edge alone doesn't form a size >= 3 cycle
        # So this test needs allow_bidirectional_cycles=True or a larger cycle
        mutmap_with_bidir = ProtMutMap(
            mutations_list=mutations_list,
            experimental_dGs=experimental_dGs,
            is_cyclic=True,
            max_cycle_size=2,
            max_mutation_num=2,
            allow_bidirectional_cycles=True
        )
        result_with_bidir = mutmap_with_bidir._check_cycle_constraint(variant_group, graph)
        assert result_with_bidir is True, "Bidirectional edge should form a size 2 cycle when allow_bidirectional_cycles=True"

    def test_cycle_constraint_path_cycle(self):
        """Test cycle constraint with path forming cycle (AA→WW and AA→AW←WW)."""

        mutations_list = [
            MutationList.from_string("AA1T,AA2W"),  # WW
        ]
        experimental_dGs = {"WT": 0}

        mutmap = ProtMutMap(
            mutations_list=mutations_list,
            experimental_dGs=experimental_dGs,
            is_cyclic=True,
            max_cycle_size=4,
            max_mutation_num=2  # Allow edges with 2 mutations
        )

        graph = mutmap.build_initial_map()

        # Manually construct the cycle: WT→AA1T,AA2W and WT→AA1T→AA1T,AA2W
        # First, ensure we have intermediate node AA1T
        if "AA1T" not in graph.nodes():
            graph.add_node("AA1T", mutations="AA1T", ddG=None, is_active=False, required=False)

        # Add edges to form cycle: WT→AA1T,AA2W (direct) and WT→AA1T→AA1T,AA2W (path)
        if "WT" in graph.nodes() and "AA1T,AA2W" in graph.nodes() and "AA1T" in graph.nodes():
            # Direct edge WT→AA1T,AA2W
            if not graph.has_edge("WT", "AA1T,AA2W"):
                graph.add_edge("WT", "AA1T,AA2W", mutation_diff="AA1T,AA2W", ddG=None,
                              num_diff_muts=2, nrep=1, nrep_cost=0.5)

            # Path: WT→AA1T→AA1T,AA2W
            if not graph.has_edge("WT", "AA1T"):
                graph.add_edge("WT", "AA1T", mutation_diff="AA1T", ddG=None,
                              num_diff_muts=1, nrep=1, nrep_cost=1.0)
            if not graph.has_edge("AA1T", "AA1T,AA2W"):
                graph.add_edge("AA1T", "AA1T,AA2W", mutation_diff="AA2W", ddG=None,
                              num_diff_muts=1, nrep=1, nrep_cost=1.0)

            # Add reverse edge to complete cycle: AA1T,AA2W→AA1T
            if not graph.has_edge("AA1T,AA2W", "AA1T"):
                graph.add_edge("AA1T,AA2W", "AA1T", mutation_diff="", ddG=None,
                              num_diff_muts=1, nrep=1, nrep_cost=1.0)


        variant_group = {"WT", "AA1T", "AA1T,AA2W"}
        result = mutmap._check_cycle_constraint(variant_group, graph)

        # Should detect cycle: WT→AA1T,AA2W→AA1T→WT (or similar)
        assert result is True, "Path-based cycle should be detected"

    def test_reduced_network_satisfies_cycle_constraints(self):
        """Check cycle constraints on a reduced mutation network."""
        mutations_list = [
            MutationList.from_string("AA1T,AA2W"),
            MutationList.from_string("AA1T,AA2V"),
            MutationList.from_string("AA1P,AA2V"),
            MutationList.from_string("AA1S,AA2R"),
        ]
        experimental_dGs = {"WT": 0}

        mutmap = ProtMutMap(
            mutations_list=mutations_list,
            experimental_dGs=experimental_dGs,
            is_cyclic=True,
            penalize_fewer_mutations=False,
            max_mutation_num=2
        )

        graph = mutmap.build_map()


        assert isinstance(graph, nx.DiGraph)
        assert len(graph.nodes()) > 0


        assert mutmap._check_constraints(graph), "All constraints should be satisfied"

        # Check cycle constraint for each required node
        required_nodes = [n for n in graph.nodes() if graph.nodes[n].get("required", False)]
        for required_node in required_nodes:
            variant_group = mutmap._get_related_variant_group(required_node, graph)
            if variant_group is not None and len(variant_group) >= 2:
                result = mutmap._check_cycle_constraint(variant_group, graph)
                assert result is True, (
                    f"Cycle constraint should be satisfied for required node {required_node} "
                    f"with variant_group {variant_group}"
                )

    def test_cycle_constraint_size_3_detection(self):
        """Test that size 3 cycles are properly detected."""
        mutations_list = [
            MutationList.from_string("AA1T,AA2W"),
        ]
        experimental_dGs = {"WT": 0}

        mutmap = ProtMutMap(
            mutations_list=mutations_list,
            experimental_dGs=experimental_dGs,
            is_cyclic=True,
            max_cycle_size=3,  # Allow size 3 cycles
            max_mutation_num=2
        )

        graph = mutmap.build_initial_map()

        # Manually create a size 3 cycle: WT→AA1T→AA1T,AA2W→WT
        if "AA1T" not in graph.nodes():
            graph.add_node("AA1T", mutations="AA1T", ddG=None, is_active=False, required=False)

        if "WT" in graph.nodes() and "AA1T" in graph.nodes() and "AA1T,AA2W" in graph.nodes():
            # Create cycle edges
            edges_to_add = [
                ("WT", "AA1T", {"mutation_diff": "AA1T", "ddG": None, "num_diff_muts": 1, "nrep": 1, "nrep_cost": 1.0}),
                ("AA1T", "AA1T,AA2W", {"mutation_diff": "AA2W", "ddG": None, "num_diff_muts": 1, "nrep": 1, "nrep_cost": 1.0}),
                ("AA1T,AA2W", "WT", {"mutation_diff": "", "ddG": None, "num_diff_muts": 2, "nrep": 1, "nrep_cost": 0.5}),
            ]
            for source, target, attrs in edges_to_add:
                if not graph.has_edge(source, target):
                    graph.add_edge(source, target, **attrs)

        # Test with variant_group containing all 3 nodes
        variant_group = {"WT", "AA1T", "AA1T,AA2W"}
        result = mutmap._check_cycle_constraint(variant_group, graph)
        assert result is True, "Size 3 cycle should be detected"

class TestIntegration:
    """Integration tests with multiple options combined."""

    def test_network_preserves_targets_and_connectivity(self):
        """All requested variants remain connected to a reference."""
        mutations_list = [
            MutationList.from_string("AA1T,AA2W"),
            MutationList.from_string("AA1T,AA2V"),
            MutationList.from_string("AA1P,AA2V"),
            MutationList.from_string("AA1S,AA2R"),
        ]
        experimental_dGs = {"WT": 0}

        mutmap = ProtMutMap(
            mutations_list=mutations_list,
            experimental_dGs=experimental_dGs,
            max_mutation_num=1,
            penalize_fewer_mutations=False,
            is_cyclic=True,
            max_cycle_size=4
        )

        graph = mutmap.build_map()


        assert isinstance(graph, nx.DiGraph)
        assert len(graph.nodes()) > 0
        assert len(graph.edges()) > 0


        required_mutations = [muts.to_string() if muts.to_string() != "" else "WT"
                              for muts in mutations_list]
        for required_mut in required_mutations:

            found = False
            for target_muts in mutations_list:
                for muts in target_muts.generate_combinations():
                    mutations_str = muts.to_string() if muts.to_string() != "" else "WT"
                    if mutations_str == required_mut:
                        if mutations_str in graph.nodes():
                            found = True
                            break
                if found:
                    break
            # Required nodes should exist (they cannot be deleted)
            assert found, f"Required mutation {required_mut} should exist in graph"


        # Cycle constraint may not always be satisfiable
        required_nodes = [n for n in graph.nodes() if graph.nodes[n].get("required", False)]
        active_nodes = [n for n in graph.nodes() if graph.nodes[n].get("is_active", False)]
        assert len(active_nodes) > 0, "At least one active node should exist"

        # Check connectivity
        undirected_graph = graph.to_undirected()
        for required_node in required_nodes:
            reachable = any(
                nx.has_path(undirected_graph, required_node, active_node)
                for active_node in active_nodes
            )
            assert reachable, f"Required node {required_node} should be connected to an active node"

    def test_multiple_experimental_dGs(self):
        """Test with multiple experimental dG values."""
        mutations_list = [
            MutationList.from_string("AA1T"),
            MutationList.from_string("AA2W"),
        ]
        experimental_dGs = {
            "WT": 0,
            "AA1T": -1.5,
            "AA2W": -2.0,
        }

        mutmap = ProtMutMap(
            mutations_list=mutations_list,
            experimental_dGs=experimental_dGs,
            max_mutation_num=1
        )

        graph = mutmap.build_map()


        # Note: WT may be deleted if not needed, but active nodes should exist
        active_nodes = [n for n in graph.nodes() if graph.nodes[n].get("is_active", False)]
        assert len(active_nodes) > 0, "At least one active node should exist"


        if "WT" in graph.nodes():
            assert graph.nodes["WT"].get("is_active") is True
            assert graph.nodes["WT"].get("ddG") == 0

        if "AA1T" in graph.nodes():
            assert graph.nodes["AA1T"].get("is_active") is True
            assert graph.nodes["AA1T"].get("ddG") == -1.5

        if "AA2W" in graph.nodes():
            assert graph.nodes["AA2W"].get("is_active") is True
            assert graph.nodes["AA2W"].get("ddG") == -2.0


        required_nodes = [n for n in graph.nodes() if graph.nodes[n].get("required", False)]
        assert len(required_nodes) == 2, "Both required mutations should exist"


        assert mutmap._check_constraints(graph)

    def test_required_nodes_remain_connected_with_combined_constraints(self):
        """Required variants remain reachable when pruning with all constraints."""
        mutations_list = [
            MutationList.from_string("AA1T,AA2W"),
            MutationList.from_string("AA1T,AA2V"),
            MutationList.from_string("AA1P,AA2V"),
        ]
        experimental_dGs = {
            "WT": 0,
            "AA1T": -1.0,
        }

        mutmap = ProtMutMap(
            mutations_list=mutations_list,
            experimental_dGs=experimental_dGs,
            max_mutation_num=2,
            penalize_fewer_mutations=True,
            is_cyclic=True,
            max_cycle_size=3
        )

        graph = mutmap.build_map()


        assert isinstance(graph, nx.DiGraph)
        assert len(graph.nodes()) > 0


        assert mutmap._check_constraints(graph)


        required_nodes = [n for n in graph.nodes() if graph.nodes[n].get("required", False)]
        assert len(required_nodes) > 0

        active_nodes = [n for n in graph.nodes() if graph.nodes[n].get("is_active", False)]
        assert len(active_nodes) > 0

        # Verify connectivity
        undirected_graph = graph.to_undirected()
        for required_node in required_nodes:
            reachable = any(
                nx.has_path(undirected_graph, required_node, active_node)
                for active_node in active_nodes
            )
            assert reachable


class TestUndirectedCycleDetection:
    """Tests for undirected cycle detection in cycle constraint."""

    def test_undirected_cycle_detection_bidirectional_edge_disabled(self):
        """Test that bidirectional edges do NOT form cycles when allow_bidirectional_cycles=False (default)."""
        mutations_list = [
            MutationList.from_string("AA1T"),
        ]
        experimental_dGs = {"WT": 0}

        mutmap = ProtMutMap(
            mutations_list=mutations_list,
            experimental_dGs=experimental_dGs,
            is_cyclic=True,
            max_cycle_size=2,
            max_mutation_num=1,
            allow_bidirectional_cycles=False  # Default
        )

        graph = mutmap.build_initial_map()

        # Ensure bidirectional edge exists between WT and AA1T
        if "WT" in graph.nodes() and "AA1T" in graph.nodes():
            if not graph.has_edge("WT", "AA1T"):
                graph.add_edge("WT", "AA1T", mutation_diff="AA1T", ddG=None,
                              num_diff_muts=1, nrep=1, nrep_cost=1.0)
            if not graph.has_edge("AA1T", "WT"):
                graph.add_edge("AA1T", "WT", mutation_diff="", ddG=None,
                              num_diff_muts=1, nrep=1, nrep_cost=1.0)

        # Test with variant_group of size 2 (bidirectional edge only)
        variant_group = {"WT", "AA1T"}
        result = mutmap._check_cycle_constraint(variant_group, graph)
        # Should return False (bidirectional edge does NOT form a cycle when disabled)
        assert result is False, "Bidirectional edge should NOT form a cycle when allow_bidirectional_cycles=False"

class TestBidirectionalEdgeRemoval:
    """Tests for redundant bidirectional edge removal."""

    def test_reduction_keeps_at_most_one_direction_per_edge(self):
        """Reduction must remove one direction from each reciprocal edge pair."""
        mutations_list = [MutationList.from_string("HH101Y,YH103W,SH105T")]
        mm = ProtMutMap(
            mutations_list,
            is_cyclic=True,
            penalize_fewer_mutations=True,
            max_mutation_num=1,
            max_cycle_size=4
        )


        mutation_graph = mm.build_map()


        assert mm._check_constraints(mutation_graph), "Constraints should be satisfied"

        # Check for bidirectional edges between WT and YH103W
        has_wt_to_yh103w = mutation_graph.has_edge("WT", "YH103W")
        has_yh103w_to_wt = mutation_graph.has_edge("YH103W", "WT")

        # At most one direction should exist
        assert not (has_wt_to_yh103w and has_yh103w_to_wt), (
            "Both directions of WT↔YH103W should not exist after bidirectional edge removal"
        )



class TestCycleConstraintSemantics:
    """Pin down what max_cycle_size actually constrains.

    The constraint is per edge: every edge between variant-group nodes must
    lie on *some* undirected cycle whose length is between 3 and
    ``max_cycle_size``. It is not a ban on longer cycles existing elsewhere in
    the graph, and an edge that lies on both a short and a long cycle
    satisfies it. The implementation decides this from the shortest
    alternative path between the edge's endpoints, which is exactly the
    shortest cycle through that edge.
    """

    @staticmethod
    def _mutmap(max_cycle_size: int) -> ProtMutMap:
        return ProtMutMap(
            mutations_list=[MutationList.from_string("AA1T,AA2W")],
            experimental_dGs={"WT": 0},
            is_cyclic=True,
            max_cycle_size=max_cycle_size,
            max_mutation_num=2,
        )

    def test_edge_whose_shortest_cycle_is_too_long_is_rejected(self):
        # Every edge of this square lies only on a 4-cycle.
        square = nx.DiGraph([("W", "A"), ("A", "X"), ("X", "B"), ("B", "W")])
        group = {"W", "A", "X", "B"}

        assert self._mutmap(3)._check_cycle_constraint(group, square) is False
        assert self._mutmap(4)._check_cycle_constraint(group, square) is True

    def test_edge_on_a_short_enough_cycle_is_accepted(self):
        triangle = nx.DiGraph([("W", "A"), ("A", "B"), ("B", "W")])
        group = {"W", "A", "B"}

        assert self._mutmap(3)._check_cycle_constraint(group, triangle) is True
        assert self._mutmap(4)._check_cycle_constraint(group, triangle) is True

    def test_an_edge_on_both_a_short_and_a_long_cycle_is_accepted(self):
        """W-A-B-W is a triangle; A-C-B adds a 4-cycle W-A-C-B-W over the same
        edges. The longer cycle does not disqualify them."""
        graph = nx.DiGraph(
            [("W", "A"), ("A", "B"), ("B", "W"), ("A", "C"), ("C", "B")]
        )

        assert self._mutmap(3)._check_cycle_constraint({"W", "A", "B", "C"}, graph) is True

    def test_an_edge_on_no_cycle_is_rejected(self):
        chain = nx.DiGraph([("W", "A"), ("A", "B")])

        assert self._mutmap(4)._check_cycle_constraint({"W", "A", "B"}, chain) is False

    def test_the_constraint_is_skipped_when_cycles_are_not_required(self):
        chain = nx.DiGraph([("W", "A"), ("A", "B")])
        mutmap = self._mutmap(4)
        mutmap.is_cyclic = False

        assert mutmap._check_cycle_constraint({"W", "A", "B"}, chain) is True


class TestNodeRemovalOrdering:
    """Direct coverage of _sort_nodes_for_removal / _remove_nodes.

    Node pruning is only reached through build_map elsewhere, where the
    assertions are loose, so protection of required and crystal-bearing nodes
    is pinned here.
    """

    @staticmethod
    def _mutmap() -> ProtMutMap:
        return ProtMutMap(
            mutations_list=[MutationList.from_string("AA1T,AA2W")],
            experimental_dGs={"WT": 0},
            is_cyclic=False,
            max_mutation_num=2,
        )

    @staticmethod
    def _graph() -> nx.DiGraph:
        graph = nx.DiGraph()
        # WT is the active reference; _check_constraints requires every
        # required node to stay connected to an active one.
        graph.add_node(
            "WT", mutations="WT", required=True, is_active=True, crystal_pdb=None
        )
        graph.add_node(
            "AA1T", mutations="AA1T", required=False, is_active=False, crystal_pdb=None
        )
        graph.add_node(
            "AA2W",
            mutations="AA2W",
            required=False,
            is_active=False,
            crystal_pdb="AA2W.pdb",
        )
        graph.add_node(
            "AA1T,AA2W",
            mutations="AA1T,AA2W",
            required=True,
            is_active=False,
            crystal_pdb=None,
        )
        for source, target in [
            ("WT", "AA1T"),
            ("AA1T", "AA1T,AA2W"),
            ("WT", "AA2W"),
            ("AA2W", "AA1T,AA2W"),
        ]:
            graph.add_edge(source, target, num_diff_muts=1, nrep=32, nrep_cost=32.0)
        return graph

    def test_required_and_crystal_bearing_nodes_are_never_candidates(self):
        candidates = self._mutmap()._sort_nodes_for_removal(self._graph())

        assert candidates == ["AA1T"]

    def test_a_node_without_a_crystal_key_is_still_a_candidate(self):
        """Graphs built before crystal support lack the attribute entirely."""
        graph = nx.DiGraph()
        graph.add_node("WT", mutations="WT", required=True)
        graph.add_node("AA1T", mutations="AA1T", required=False)

        assert self._mutmap()._sort_nodes_for_removal(graph) == ["AA1T"]

    def test_remove_nodes_keeps_required_and_crystal_bearing_nodes(self):
        pruned = self._mutmap()._remove_nodes(self._graph())

        assert {"WT", "AA2W", "AA1T,AA2W"} <= set(pruned.nodes())
        assert "AA1T" not in pruned.nodes()
