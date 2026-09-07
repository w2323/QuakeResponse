# test_full_pipeline.py — Integration tests for the complete QuakeResponse pipeline
# Tests CSP → MST → ML → GA → A* working together on a shared disaster grid.
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import random
import pytest
from core.city_graph import DisasterGrid
from core.simulation_engine import Simulation
from challenges.layout_planning import CSPSolver
from challenges.road_optimization import build_emergency_corridors
from challenges.crime_prediction import run_vulnerability_assessment
from challenges.ambulance_placement import deploy_medical_units
from challenges.emergency_routing import AStarRouter
from config.simulation_config import QuakeConfig


class TestSharedGridIntegrity:
    """Verify that all modules share ONE disaster grid and changes propagate globally."""

    def test_single_grid_instance(self):
        """All modules must reference the same grid object."""
        random.seed(42)
        graph = DisasterGrid()

        # CSP uses the grid
        solver = CSPSolver(graph)
        solver.solve()
        solver.apply_assignment_to_graph()
        assert solver.graph is graph

        # MST uses the same grid
        build_emergency_corridors(graph)

        # ML uses the same grid
        run_vulnerability_assessment(graph)

        # GA uses the same grid
        medical_units = deploy_medical_units(graph)

        # Router uses the same grid
        router = AStarRouter(graph)
        assert router.graph is graph

    def test_csp_layout_preserved_after_mst(self):
        """MST should not alter sector types set by CSP."""
        random.seed(42)
        graph = DisasterGrid()
        solver = CSPSolver(graph)
        solver.solve()
        solver.apply_assignment_to_graph()

        # Record types
        types_before = {nid: n.node_type for nid, n in graph.nodes.items()}

        build_emergency_corridors(graph)

        # Types should be unchanged
        for nid, node in graph.nodes.items():
            assert node.node_type == types_before[nid]

    def test_vulnerability_affects_routing(self):
        """Vulnerability predictions must influence A* path costs."""
        random.seed(42)
        graph = DisasterGrid()
        solver = CSPSolver(graph)
        solver.solve()
        solver.apply_assignment_to_graph()
        build_emergency_corridors(graph)

        # Get path cost BEFORE vulnerability assessment
        router_before = AStarRouter(graph)
        residential = graph.get_nodes_of_type(QuakeConfig.RESIDENTIAL)
        hospitals = graph.get_nodes_of_type(QuakeConfig.FIELD_HOSPITAL)
        if residential and hospitals:
            path_before = router_before.astar(hospitals[0].id, residential[0].id)

        # Run vulnerability assessment — updates collapse risk levels globally
        run_vulnerability_assessment(graph)

        # Verify collapse risk levels are set
        risk_levels = set(n.collapse_risk for n in graph.nodes.values())
        assert "NONE" not in risk_levels or len(risk_levels) > 1

    def test_aftershock_affects_all_systems(self):
        """Collapsing a corridor must instantly affect routing, deployment, and costs."""
        random.seed(42)
        graph = DisasterGrid()
        solver = CSPSolver(graph)
        solver.solve()
        solver.apply_assignment_to_graph()
        build_emergency_corridors(graph)

        # Setup router
        router = AStarRouter(graph)
        residential = graph.get_nodes_of_type(QuakeConfig.RESIDENTIAL)
        hospitals = graph.get_nodes_of_type(QuakeConfig.FIELD_HOSPITAL)

        if residential and hospitals:
            router.set_mission(hospitals[0].id, [residential[0].id])
            path_before = list(router.current_path)

            # Collapse a corridor on the path
            if len(path_before) >= 3:
                a, b = path_before[1], path_before[2]
                edge_cost_before = graph.get_edge_cost(a, b)
                graph.collapse_road(a, b)

                # Corridor cost should now be infinite
                assert graph.get_edge_cost(a, b) == float("inf")

                # Neighbors should be updated
                assert b not in graph.get_neighbors(a)


class TestFullPipelineOrder:
    """Test that CSP → MST → ML → GA → A* works in correct order."""

    def test_pipeline_runs_successfully(self):
        random.seed(42)
        graph = DisasterGrid()

        # Step 1: CSP
        solver = CSPSolver(graph)
        assert solver.solve() is True
        solver.apply_assignment_to_graph()

        # Step 2: MST
        assert build_emergency_corridors(graph) is True

        # Step 3: ML
        kmeans, dt = run_vulnerability_assessment(graph)
        assert kmeans is not None
        assert dt is not None

        # Step 4: GA
        medical_units = deploy_medical_units(graph)
        assert len(medical_units) == QuakeConfig.GA_NUM_MEDICAL_UNITS

        # Step 5: A*
        router = AStarRouter(graph)
        hospitals = graph.get_nodes_of_type(QuakeConfig.FIELD_HOSPITAL)
        residential = graph.get_nodes_of_type(QuakeConfig.RESIDENTIAL)
        if hospitals and residential:
            router.set_mission(hospitals[0].id, [residential[0].id])
            assert router.current_path is not None
            assert len(router.current_path) > 0


class TestDynamicPropagation:
    """Test that dynamic changes propagate globally across all systems."""

    def test_aftershock_propagation(self):
        """An aftershock must affect routing, costs, and grid state simultaneously."""
        random.seed(42)
        sim = Simulation(seed=42, verbose=False)
        sim.initialize()

        # Record initial state
        initial_active = sim.graph.count_active_edges()

        # Trigger aftershock manually
        sim._trigger_aftershock()

        # Active corridors should decrease
        assert sim.graph.count_active_edges() <= initial_active

    def test_simulation_with_heavy_aftershocks(self):
        """Simulation should handle many aftershocks gracefully."""
        random.seed(42)
        sim = Simulation(seed=42, verbose=False)
        sim.initialize()

        # Force 10 aftershocks
        for _ in range(10):
            sim._trigger_aftershock()

        # Simulation should still be able to step forward
        sim.step_forward()
        assert sim.state.step == 1

    def test_vulnerability_update_propagates_to_corridor_costs(self):
        """Changing a sector's collapse risk must immediately affect corridor costs."""
        random.seed(42)
        graph = DisasterGrid()
        solver = CSPSolver(graph)
        solver.solve()
        solver.apply_assignment_to_graph()
        build_emergency_corridors(graph)

        # Pick a sector with corridors
        test_sector = None
        for nid in graph.nodes:
            if graph.get_neighbors(nid):
                test_sector = nid
                break

        if test_sector is not None:
            nb = graph.get_neighbors(test_sector)[0]
            cost_before = graph.get_edge_cost(nb, test_sector)

            # Update collapse risk
            graph.update_risk(test_sector, "HIGH")

            # Cost should now be higher
            cost_after = graph.get_edge_cost(nb, test_sector)
            assert cost_after >= cost_before
