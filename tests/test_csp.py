# test_csp.py — Unit tests for Challenge 1: Emergency Resource Zone Planning (CSP)
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import random
import pytest
from core.city_graph import DisasterGrid
from challenges.layout_planning import CSPSolver
from config.simulation_config import QuakeConfig


@pytest.fixture
def seeded_graph():
    random.seed(42)
    return DisasterGrid()


class TestCSPSolver:
    def test_solve_succeeds(self, seeded_graph):
        solver = CSPSolver(seeded_graph)
        assert solver.solve() is True

    def test_c1_hazard_zone_not_adjacent_to_shelter_hospital(self, seeded_graph):
        """C1: No Hazard Zone adjacent to Shelter or Field Hospital."""
        solver = CSPSolver(seeded_graph)
        solver.solve()
        solver.apply_assignment_to_graph()
        for nid, node in seeded_graph.nodes.items():
            if node.node_type == QuakeConfig.HAZARD_ZONE:
                for nb_id in seeded_graph.get_grid_neighbors(nid):
                    nb_type = seeded_graph.nodes[nb_id].node_type
                    assert nb_type not in (QuakeConfig.FIELD_HOSPITAL, QuakeConfig.SHELTER), \
                        f"C1 violation: HazardZone({nid}) adjacent to {nb_type}({nb_id})"

    def test_c2_residential_within_3_blocks_of_hospital(self, seeded_graph):
        """C2: Every Residential sector within 3 grid blocks of a Field Hospital."""
        solver = CSPSolver(seeded_graph)
        solver.solve()
        solver.apply_assignment_to_graph()
        hospitals = [n.id for n in seeded_graph.get_nodes_of_type(QuakeConfig.FIELD_HOSPITAL)]
        assert len(hospitals) > 0
        for node in seeded_graph.get_nodes_of_type(QuakeConfig.RESIDENTIAL):
            min_hops = min(solver._bfs_grid_hops(node.id, h) for h in hospitals)
            assert 0 <= min_hops <= QuakeConfig.HOSPITAL_HOP_LIMIT, \
                f"C2 violation: Residential({node.id}) is {min_hops} blocks from field hospital"

    def test_c3_generator_within_2_blocks_of_hazard_zone(self, seeded_graph):
        """C3: Every Generator Station within 2 grid blocks of a Hazard Zone."""
        solver = CSPSolver(seeded_graph)
        solver.solve()
        solver.apply_assignment_to_graph()
        hazard_zones = [n.id for n in seeded_graph.get_nodes_of_type(QuakeConfig.HAZARD_ZONE)]
        assert len(hazard_zones) > 0
        for node in seeded_graph.get_nodes_of_type(QuakeConfig.GENERATOR_STATION):
            min_hops = min(solver._bfs_grid_hops(node.id, i) for i in hazard_zones)
            assert 0 <= min_hops <= QuakeConfig.HAZARD_HOP_LIMIT, \
                f"C3 violation: GeneratorStation({node.id}) is {min_hops} blocks from hazard zone"

    def test_required_counts_met(self, seeded_graph):
        """All required zone types placed in sufficient quantities."""
        solver = CSPSolver(seeded_graph)
        solver.solve()
        solver.apply_assignment_to_graph()
        from collections import Counter
        counts = Counter(n.node_type for n in seeded_graph.nodes.values())
        for node_type, required in QuakeConfig.REQUIRED_COUNTS.items():
            assert counts.get(node_type, 0) >= required, \
                f"Required {required} {node_type}, got {counts.get(node_type, 0)}"

    def test_population_assigned(self, seeded_graph):
        """Population values are set after layout."""
        solver = CSPSolver(seeded_graph)
        solver.solve()
        solver.apply_assignment_to_graph()
        res_nodes = seeded_graph.get_nodes_of_type(QuakeConfig.RESIDENTIAL)
        assert len(res_nodes) > 0
        for node in res_nodes:
            assert node.population >= 50

    def test_identify_failing_constraint(self, seeded_graph):
        """Failing constraint identification returns a meaningful string."""
        solver = CSPSolver(seeded_graph)
        result = solver.identify_failing_constraint()
        assert isinstance(result, str)
        assert "CONFLICT" in result

    def test_mrv_selection(self, seeded_graph):
        """MRV heuristic returns a valid unassigned variable."""
        solver = CSPSolver(seeded_graph)
        for nid in seeded_graph.nodes:
            solver.domains[nid] = list(QuakeConfig.ALL_TYPES)
        var = solver.select_unassigned_variable()
        assert var is not None
        assert var in seeded_graph.nodes

    def test_forward_checking_prunes_domains(self, seeded_graph):
        """Forward checking reduces neighbor domains after assignment."""
        solver = CSPSolver(seeded_graph)
        for nid in seeded_graph.nodes:
            solver.domains[nid] = list(QuakeConfig.ALL_TYPES)
        # Assign HAZARD_ZONE to sector 0
        solver._assign(0, QuakeConfig.HAZARD_ZONE)
        # Neighbors of 0 should NOT have FIELD_HOSPITAL or SHELTER in domain
        for nb in seeded_graph.get_grid_neighbors(0):
            assert QuakeConfig.FIELD_HOSPITAL not in solver.domains[nb]
            assert QuakeConfig.SHELTER not in solver.domains[nb]
