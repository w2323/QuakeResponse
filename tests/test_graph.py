# test_graph.py — Unit tests for DisasterGrid, Node, Edge
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import random
import pytest
from core.city_graph import DisasterGrid, Node, Edge
from config.simulation_config import QuakeConfig


class TestNodeCreation:
    def test_node_attributes(self):
        node = Node(3, 5)
        assert node.row == 3
        assert node.col == 5
        assert node.id == 3 * QuakeConfig.GRID_COLS + 5
        assert node.node_type == QuakeConfig.EMPTY
        assert node.collapse_risk == "NONE"
        assert node.population == 0
        assert node.accessible is True

    def test_node_serialization(self):
        node = Node(2, 4)
        node.node_type = QuakeConfig.FIELD_HOSPITAL
        d = node.to_dict()
        assert d["node_type"] == QuakeConfig.FIELD_HOSPITAL
        assert d["row"] == 2
        assert d["col"] == 4


class TestEdgeCreation:
    def test_edge_attributes(self):
        edge = Edge(10, 11, 1.0)
        assert edge.node_a == 10
        assert edge.node_b == 11
        assert edge.cost == 1.0
        assert edge.blocked is False

    def test_edge_blocked_cost(self):
        graph = DisasterGrid()
        edge = Edge(0, 1, 1.0)
        edge.blocked = True
        assert edge.effective_cost(graph) == float("inf")

    def test_edge_risk_multiplier(self):
        graph = DisasterGrid()
        graph.nodes[1].collapse_risk = "HIGH"
        edge = Edge(0, 1, 1.0)
        assert edge.effective_cost(graph) == 2.0  # RISK_MULTIPLIER["HIGH"] = 2.0


class TestDisasterGrid:
    def test_grid_initialization(self):
        graph = DisasterGrid()
        total = QuakeConfig.GRID_ROWS * QuakeConfig.GRID_COLS
        assert len(graph.nodes) == total
        assert len(graph.adj) == total
        assert len(graph.edges) == 0

    def test_add_edge(self):
        graph = DisasterGrid()
        graph.add_edge(0, 1, 1.0)
        assert (0, 1) in graph.edges
        assert 1 in graph.adj[0]
        assert 0 in graph.adj[1]

    def test_add_duplicate_edge(self):
        graph = DisasterGrid()
        graph.add_edge(0, 1, 1.0)
        graph.add_edge(0, 1, 1.0)
        assert len(graph.edges) == 1

    def test_collapse_road(self):
        graph = DisasterGrid()
        graph.add_edge(0, 1, 1.0)
        graph.collapse_road(0, 1)
        assert graph.edges[(0, 1)].blocked is True
        assert 1 not in graph.adj[0]
        assert 0 not in graph.adj[1]

    def test_double_collapse_safe(self):
        """Collapsing an already-collapsed road should not raise errors."""
        graph = DisasterGrid()
        graph.add_edge(0, 1, 1.0)
        graph.collapse_road(0, 1)
        graph.collapse_road(0, 1)  # Should not error
        assert graph.edges[(0, 1)].blocked is True

    def test_get_neighbors_excludes_collapsed(self):
        graph = DisasterGrid()
        graph.add_edge(0, 1, 1.0)
        graph.add_edge(0, 15, 1.0)  # Down neighbor in 15-col grid
        graph.collapse_road(0, 1)
        neighbors = graph.get_neighbors(0)
        assert 1 not in neighbors
        assert 15 in neighbors

    def test_get_edge_cost_collapsed(self):
        graph = DisasterGrid()
        graph.add_edge(0, 1, 1.0)
        graph.collapse_road(0, 1)
        assert graph.get_edge_cost(0, 1) == float("inf")

    def test_get_edge_cost_with_risk(self):
        graph = DisasterGrid()
        graph.add_edge(0, 1, 1.0)
        graph.update_risk(1, "MEDIUM")
        cost = graph.get_edge_cost(0, 1)
        assert cost == 1.5  # 1.0 * 1.5

    def test_observer_notification(self):
        graph = DisasterGrid()
        graph.add_edge(0, 1, 1.0)
        notified = []
        graph.register_observer(lambda a, b: notified.append((a, b)))
        graph.collapse_road(0, 1)
        assert len(notified) == 1
        assert notified[0] == (0, 1)

    def test_dijkstra_basic(self):
        graph = DisasterGrid()
        graph.add_edge(0, 1, 1.0)
        graph.add_edge(1, 2, 1.0)
        dist = graph.dijkstra(0)
        assert dist[0] == 0.0
        assert dist[1] == 1.0
        assert dist[2] == 2.0

    def test_dijkstra_collapsed_path(self):
        graph = DisasterGrid()
        graph.add_edge(0, 1, 1.0)
        graph.add_edge(1, 2, 1.0)
        graph.collapse_road(0, 1)
        dist = graph.dijkstra(0)
        assert dist[1] == float("inf")
        assert dist[2] == float("inf")

    def test_bfs_distance(self):
        graph = DisasterGrid()
        graph.add_edge(0, 1, 1.0)
        graph.add_edge(1, 2, 1.0)
        assert graph.bfs_distance(0, 2) == 2
        assert graph.bfs_distance(0, 0) == 0

    def test_bfs_no_path(self):
        graph = DisasterGrid()
        assert graph.bfs_distance(0, 100) == -1

    def test_get_grid_neighbors(self):
        graph = DisasterGrid()
        # Corner sector (0,0) should have 2 grid neighbors
        nb = graph.get_grid_neighbors(0)
        assert len(nb) == 2
        # Center-ish sector should have 4
        center_id = 7 * QuakeConfig.GRID_COLS + 7
        nb = graph.get_grid_neighbors(center_id)
        assert len(nb) == 4

    def test_serialization_roundtrip(self):
        random.seed(42)
        graph = DisasterGrid()
        graph.add_edge(0, 1, 1.0)
        graph.add_edge(1, 2, 1.5)
        graph.nodes[0].node_type = QuakeConfig.FIELD_HOSPITAL
        graph.update_risk(1, "HIGH")
        d = graph.to_dict()
        graph2 = DisasterGrid.from_dict(d)
        assert graph2.nodes[0].node_type == QuakeConfig.FIELD_HOSPITAL
        assert graph2.nodes[1].collapse_risk == "HIGH"
        assert len(graph2.edges) == 2

    def test_count_edges(self):
        graph = DisasterGrid()
        graph.add_edge(0, 1, 1.0)
        graph.add_edge(1, 2, 1.0)
        graph.collapse_road(0, 1)
        assert graph.count_active_edges() == 1
        assert graph.count_blocked_edges() == 1

    def test_populate_after_layout(self):
        random.seed(42)
        graph = DisasterGrid()
        graph.nodes[0].node_type = QuakeConfig.RESIDENTIAL
        graph.nodes[1].node_type = QuakeConfig.FIELD_HOSPITAL
        graph.populate_after_layout()
        assert 50 <= graph.nodes[0].population <= 200
        assert graph.nodes[1].population == 20
