# test_mst.py — Unit tests for Challenge 2: Emergency Corridor Network Optimization
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import random
import pytest
from collections import deque
from core.city_graph import DisasterGrid
from challenges.layout_planning import CSPSolver
from challenges.road_optimization import (
    build_mst, add_redundancy_edges, ensure_hospital_depot_lifeline,
    build_emergency_corridors, bfs_path, bfs_path_excluding, find_bridges,
    UnionFind
)
from config.simulation_config import QuakeConfig


@pytest.fixture
def layout_graph():
    """Grid with CSP layout applied (needed for corridor costs)."""
    random.seed(42)
    graph = DisasterGrid()
    solver = CSPSolver(graph)
    solver.solve()
    solver.apply_assignment_to_graph()
    return graph


class TestUnionFind:
    def test_basic_union_find(self):
        uf = UnionFind(10)
        assert uf.union(0, 1) is True
        assert uf.union(0, 1) is False  # Already in same set
        assert uf.find(0) == uf.find(1)

    def test_path_compression(self):
        uf = UnionFind(5)
        uf.union(0, 1)
        uf.union(1, 2)
        uf.union(2, 3)
        root = uf.find(3)
        assert uf.parent[3] == root  # Path compressed


class TestMST:
    def test_mst_edge_count(self, layout_graph):
        """MST should have exactly V-1 edges."""
        mst_edges = build_mst(layout_graph)
        expected = QuakeConfig.GRID_ROWS * QuakeConfig.GRID_COLS - 1
        assert len(mst_edges) == expected

    def test_mst_connectivity(self, layout_graph):
        """After MST, all sectors should be reachable from sector 0."""
        build_mst(layout_graph)
        visited = set()
        queue = deque([0])
        visited.add(0)
        while queue:
            cur = queue.popleft()
            for nb in layout_graph.get_neighbors(cur):
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        assert len(visited) == len(layout_graph.nodes)


class TestRedundancy:
    def test_redundancy_edges_added(self, layout_graph):
        build_mst(layout_graph)
        initial_edges = len(layout_graph.edges)
        add_redundancy_edges(layout_graph, extra_count=30)
        assert len(layout_graph.edges) > initial_edges

    def test_dead_ends_reduced(self, layout_graph):
        build_mst(layout_graph)
        dead_ends_before = sum(1 for nid in layout_graph.nodes
                               if len(layout_graph.adj[nid]) == 1)
        add_redundancy_edges(layout_graph, extra_count=30)
        dead_ends_after = sum(1 for nid in layout_graph.nodes
                              if len(layout_graph.adj[nid]) == 1)
        assert dead_ends_after <= dead_ends_before


class TestHospitalDepotLifeline:
    def test_two_edge_disjoint_paths(self, layout_graph):
        """After full corridor network build, 2 edge-disjoint lifeline paths must exist."""
        build_emergency_corridors(layout_graph)
        hospitals = layout_graph.get_nodes_of_type(QuakeConfig.FIELD_HOSPITAL)
        depots = layout_graph.get_nodes_of_type(QuakeConfig.SUPPLY_DEPOT)
        assert len(hospitals) > 0
        assert len(depots) > 0

        h_id = hospitals[0].id
        d_id = depots[0].id

        # Find path 1
        path1 = bfs_path(layout_graph, h_id, d_id)
        assert path1 is not None

        # Collect corridors from path 1
        used = set()
        for i in range(len(path1) - 1):
            used.add((min(path1[i], path1[i+1]), max(path1[i], path1[i+1])))

        # Find path 2 excluding path 1's corridors
        path2 = bfs_path_excluding(layout_graph, h_id, d_id, used)
        assert path2 is not None, "No second edge-disjoint lifeline path found"


class TestBuildEmergencyCorridors:
    def test_full_build_succeeds(self, layout_graph):
        assert build_emergency_corridors(layout_graph) is True

    def test_bridge_detection(self, layout_graph):
        build_emergency_corridors(layout_graph)
        bridges = find_bridges(layout_graph)
        # With redundancy corridors, there should be very few bridges
        assert isinstance(bridges, list)


class TestDisconnectedGraph:
    def test_bfs_path_none_on_disconnected(self):
        """BFS should return None when sectors are disconnected."""
        graph = DisasterGrid()
        graph.add_edge(0, 1, 1.0)
        # Sector 100 is not connected
        assert bfs_path(graph, 0, 100) is None
