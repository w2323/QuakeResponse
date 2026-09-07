# test_astar.py — Unit tests for Challenge 4: Survivor Rescue Routing (A*)
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import random
import pytest
from core.city_graph import DisasterGrid
from challenges.layout_planning import CSPSolver
from challenges.road_optimization import build_emergency_corridors
from challenges.emergency_routing import AStarRouter, heuristic
from config.simulation_config import QuakeConfig


@pytest.fixture
def ready_graph():
    random.seed(42)
    graph = DisasterGrid()
    solver = CSPSolver(graph)
    solver.solve()
    solver.apply_assignment_to_graph()
    build_emergency_corridors(graph)
    return graph


class TestHeuristic:
    def test_admissibility(self):
        """Heuristic must be admissible (never overestimates)."""
        graph = DisasterGrid()
        # Add a simple path
        graph.add_edge(0, 1, 1.0)
        graph.add_edge(1, 2, 1.0)
        # Heuristic from 0 to 2 should be <= actual path cost
        h = heuristic(graph, 0, 2)
        actual_cost = 2.0
        assert h <= actual_cost

    def test_zero_at_goal(self):
        """Heuristic should be 0 when sector equals goal."""
        graph = DisasterGrid()
        assert heuristic(graph, 0, 0) == 0.0


class TestAStarRouter:
    def test_shortest_path_correctness(self, ready_graph):
        """A* should find a valid path between connected sectors."""
        router = AStarRouter(ready_graph)
        hospitals = ready_graph.get_nodes_of_type(QuakeConfig.FIELD_HOSPITAL)
        residential = ready_graph.get_nodes_of_type(QuakeConfig.RESIDENTIAL)
        if hospitals and residential:
            path = router.astar(hospitals[0].id, residential[0].id)
            assert path is not None
            assert path[0] == hospitals[0].id
            assert path[-1] == residential[0].id

    def test_path_is_connected(self, ready_graph):
        """Every consecutive pair in the path should be actual neighbors."""
        router = AStarRouter(ready_graph)
        path = router.astar(0, 100)
        if path:
            for i in range(len(path) - 1):
                neighbors = ready_graph.get_neighbors(path[i])
                assert path[i + 1] in neighbors

    def test_collapsed_corridor_rerouting(self, ready_graph):
        """Collapsing a corridor on the current path should trigger replanning."""
        router = AStarRouter(ready_graph)
        hospitals = ready_graph.get_nodes_of_type(QuakeConfig.FIELD_HOSPITAL)
        residential = ready_graph.get_nodes_of_type(QuakeConfig.RESIDENTIAL)
        if hospitals and len(residential) >= 2:
            start = hospitals[0].id
            survivors = [residential[0].id, residential[1].id]
            router.set_mission(start, survivors)
            old_path = list(router.current_path)

            # Collapse a corridor on the path (if path has at least 3 sectors)
            if len(old_path) >= 3:
                a, b = old_path[1], old_path[2]
                ready_graph.collapse_road(a, b)
                # Path should have been recalculated
                assert router.total_reroutes > 0 or router.current_path != old_path

    def test_unreachable_target(self, ready_graph):
        """A* should return None for unreachable sectors."""
        # Create a completely isolated sector by having no edges
        router = AStarRouter(ready_graph)
        # Sector 0 to a sector with no edges
        isolated_graph = DisasterGrid()
        iso_router = AStarRouter(isolated_graph)
        path = iso_router.astar(0, 100)
        assert path is None

    def test_same_sector_path(self, ready_graph):
        router = AStarRouter(ready_graph)
        path = router.astar(0, 0)
        assert path == [0]

    def test_advance_step(self, ready_graph):
        """advance_one_step should move the rescue team forward."""
        router = AStarRouter(ready_graph)
        residential = ready_graph.get_nodes_of_type(QuakeConfig.RESIDENTIAL)
        if len(residential) >= 1:
            router.set_mission(0, [residential[0].id])
            if len(router.current_path) >= 2:
                expected_next = router.current_path[1]
                result = router.advance_one_step()
                assert result == expected_next

    def test_survivor_visited_on_arrival(self, ready_graph):
        """When rescue team reaches a survivor's sector, it should be marked visited."""
        router = AStarRouter(ready_graph)
        residential = ready_graph.get_nodes_of_type(QuakeConfig.RESIDENTIAL)
        if residential:
            target = residential[0].id
            router.set_mission(0, [target])
            # Walk the entire path
            while router.current_path and len(router.current_path) >= 2:
                router.advance_one_step()
            assert target in router.visited or target not in router.survivors

    def test_serialization(self, ready_graph):
        router = AStarRouter(ready_graph)
        residential = ready_graph.get_nodes_of_type(QuakeConfig.RESIDENTIAL)
        if residential:
            router.set_mission(0, [residential[0].id])
        d = router.to_dict()
        assert "current_pos" in d
        assert "survivors_remaining" in d
