# city_graph.py — QuakeResponse shared disaster grid
# This is the foundation of the entire project. Every module imports from here.
# Contains: Node, Edge, DisasterGrid — kept together intentionally (no unnecessary fragmentation).

import heapq
import random
from collections import deque
from config.simulation_config import QuakeConfig


# ──────────────────────────────────────────────────────────
# Node — represents a single sector/block in the disaster grid
# ──────────────────────────────────────────────────────────
class Node:
    """A single grid sector in the disaster zone. Stores zone type, population, and vulnerability data."""

    def __init__(self, row: int, col: int):
        self.id: int = row * QuakeConfig.GRID_COLS + col
        self.row: int = row
        self.col: int = col
        self.node_type: str = QuakeConfig.EMPTY        # Assigned by CSP solver
        self.population: int = 0                       # Set after layout
        self.vulnerability_index: float = 0.0          # Numeric vulnerability score from ML
        self.collapse_risk: str = "NONE"               # "HIGH", "MEDIUM", "LOW", "NONE"
        self.accessible: bool = True                   # False if sector becomes isolated

    def __repr__(self) -> str:
        return f"Node({self.row},{self.col}, {self.node_type})"

    def to_dict(self) -> dict:
        """Serialize node state for API / frontend consumption."""
        return {
            "id": self.id,
            "row": self.row,
            "col": self.col,
            "node_type": self.node_type,
            "population": self.population,
            "vulnerability_index": round(self.vulnerability_index, 3),
            "collapse_risk": self.collapse_risk,
            "accessible": self.accessible,
        }


# ──────────────────────────────────────────────────────────
# Edge — represents a road/corridor between two sectors
# ──────────────────────────────────────────────────────────
class Edge:
    """A road connecting two sectors. Tracks base cost and collapsed status."""

    def __init__(self, node_a_id: int, node_b_id: int, base_cost: float = QuakeConfig.COST_STANDARD):
        self.node_a: int = node_a_id
        self.node_b: int = node_b_id
        self.cost: float = base_cost          # Base traversal cost
        self.blocked: bool = False            # True = road is impassable (collapsed)

    def effective_cost(self, graph: "DisasterGrid") -> float:
        """
        Compute actual traversal cost factoring in collapse risk multipliers.
        Returns infinity if the road is collapsed.
        """
        if self.blocked:
            return float("inf")
        risk = graph.nodes[self.node_b].collapse_risk
        return self.cost * QuakeConfig.RISK_MULTIPLIER.get(risk, 1.0)

    def other(self, node_id: int) -> int:
        """Return the other endpoint of this edge."""
        return self.node_b if node_id == self.node_a else self.node_a

    def to_dict(self) -> dict:
        """Serialize edge state for API / frontend consumption."""
        return {
            "node_a": self.node_a,
            "node_b": self.node_b,
            "cost": self.cost,
            "blocked": self.blocked,
        }


# ──────────────────────────────────────────────────────────
# DisasterGrid — the entire disaster zone's graph structure
# This is the SINGLE SHARED GRAPH for the whole system.
# No module may maintain its own copy.
# ──────────────────────────────────────────────────────────
class DisasterGrid:
    """
    Centralized disaster zone grid. All modules share ONE instance.

    - Nodes are created on init (15x15 = 225 sectors, all EMPTY)
    - Edges are added by Challenge 2 (emergency corridor network)
    - Observers are notified when roads collapse
    - Dijkstra and BFS use effective_cost which respects collapse risk multipliers
    """

    def __init__(self):
        self.nodes: dict = {}       # id -> Node
        self.edges: dict = {}       # (min_id, max_id) -> Edge (canonical key)
        self.adj: dict = {}         # id -> list of neighbor node IDs
        self._observers: list = []  # Callbacks notified when a road collapses
        self._init_grid()

    def _init_grid(self):
        """Create the grid of GRID_ROWS x GRID_COLS sectors, all EMPTY."""
        for row in range(QuakeConfig.GRID_ROWS):
            for col in range(QuakeConfig.GRID_COLS):
                node = Node(row, col)
                self.nodes[node.id] = node
                self.adj[node.id] = []

    # ── Node access ─────────────────────────────────────────

    def get_node(self, row: int, col: int) -> Node:
        """Get a node by its grid coordinates."""
        return self.nodes[row * QuakeConfig.GRID_COLS + col]

    def get_nodes_of_type(self, node_type: str) -> list:
        """Return all nodes matching the given type."""
        return [n for n in self.nodes.values() if n.node_type == node_type]

    # ── Edge management ─────────────────────────────────────

    def add_edge(self, id_a: int, id_b: int, cost: float = QuakeConfig.COST_STANDARD):
        """Add an edge between two sectors. Ignores duplicates."""
        key = (min(id_a, id_b), max(id_a, id_b))
        if key in self.edges:
            return
        edge = Edge(id_a, id_b, cost)
        self.edges[key] = edge
        self.adj[id_a].append(id_b)
        self.adj[id_b].append(id_a)

    def remove_edge(self, id_a: int, id_b: int):
        """
        Mark an edge as collapsed and remove from adjacency lists.
        The Edge object is kept (with blocked=True) for logging and history.
        Notifies all registered observers.
        """
        key = (min(id_a, id_b), max(id_a, id_b))
        if key not in self.edges:
            return
        edge = self.edges[key]
        if edge.blocked:
            return  # Already collapsed — prevent double processing
        edge.blocked = True
        # Remove from adjacency lists safely
        if id_b in self.adj[id_a]:
            self.adj[id_a].remove(id_b)
        if id_a in self.adj[id_b]:
            self.adj[id_b].remove(id_a)
        # Notify observers (A* router listens here)
        self._notify_collapsed(id_a, id_b)

    def collapse_road(self, id_a: int, id_b: int):
        """Alias for remove_edge. Called by simulation during aftershock events."""
        self.remove_edge(id_a, id_b)

    # ── Neighbor queries ────────────────────────────────────

    def get_neighbors(self, node_id: int) -> list:
        """
        Return neighbors reachable via non-collapsed edges.
        Filters adjacency list against actual edge blocked status.
        """
        result = []
        for nb_id in self.adj[node_id]:
            key = (min(node_id, nb_id), max(node_id, nb_id))
            edge = self.edges.get(key)
            if edge and not edge.blocked:
                result.append(nb_id)
        return result

    def get_grid_neighbors(self, node_id: int) -> list:
        """
        Return 4-directional grid neighbors (up/down/left/right).
        Does NOT require an edge to exist — used by CSP for constraint checking.
        """
        node = self.nodes[node_id]
        neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = node.row + dr, node.col + dc
            if 0 <= nr < QuakeConfig.GRID_ROWS and 0 <= nc < QuakeConfig.GRID_COLS:
                neighbors.append(nr * QuakeConfig.GRID_COLS + nc)
        return neighbors

    # ── Cost queries ────────────────────────────────────────

    def get_edge_cost(self, id_a: int, id_b: int) -> float:
        """
        Return the effective cost of traversing the edge.
        Factors in collapse risk multipliers. Returns infinity if collapsed or missing.
        """
        key = (min(id_a, id_b), max(id_a, id_b))
        edge = self.edges.get(key)
        if edge is None:
            return float("inf")
        return edge.effective_cost(self)

    # ── Vulnerability management ────────────────────────────

    def update_risk(self, node_id: int, collapse_risk: str, vulnerability_index: float = None):
        """
        Update a sector's collapse risk level. Called by the vulnerability assessment module.
        All subsequent get_edge_cost calls automatically use the new multiplier.
        """
        node = self.nodes[node_id]
        node.collapse_risk = collapse_risk
        if vulnerability_index is not None:
            node.vulnerability_index = vulnerability_index

    # ── Observer pattern ────────────────────────────────────

    def register_observer(self, callback):
        """Register a callback to be notified when a road collapses: callback(id_a, id_b)."""
        self._observers.append(callback)

    def _notify_collapsed(self, id_a: int, id_b: int):
        """Notify all observers of a road collapse event."""
        for cb in self._observers:
            cb(id_a, id_b)

    # ── Graph algorithms ────────────────────────────────────

    def bfs_distance(self, start_id: int, end_id: int) -> int:
        """
        BFS hop count (ignoring edge costs). Used by CSP for hop constraints.
        Returns -1 if no path exists.
        """
        if start_id == end_id:
            return 0
        visited = {start_id}
        queue = deque([(start_id, 0)])
        while queue:
            cur, dist = queue.popleft()
            for nb in self.get_neighbors(cur):
                if nb == end_id:
                    return dist + 1
                if nb not in visited:
                    visited.add(nb)
                    queue.append((nb, dist + 1))
        return -1

    def dijkstra(self, start_id: int) -> dict:
        """
        Dijkstra's shortest path from scratch using heapq.
        Returns {node_id: shortest_distance_from_start}.
        Respects collapse risk multipliers and collapsed roads via get_edge_cost.
        """
        dist = {nid: float("inf") for nid in self.nodes}
        dist[start_id] = 0.0
        heap = [(0.0, start_id)]
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist[u]:
                continue
            for v in self.get_neighbors(u):
                nd = d + self.get_edge_cost(u, v)
                if nd < dist[v]:
                    dist[v] = nd
                    heapq.heappush(heap, (nd, v))
        return dist

    # ── Post-layout population assignment ───────────────────

    def populate_after_layout(self):
        """
        After CSP assigns zone types, set population values for each sector.
        Uses POPULATION_MAP from config.
        """
        for node in self.nodes.values():
            pop_cfg = QuakeConfig.POPULATION_MAP.get(node.node_type, 0)
            if isinstance(pop_cfg, tuple):
                node.population = random.randint(pop_cfg[0], pop_cfg[1])
            else:
                node.population = pop_cfg

    # ── Graph statistics ────────────────────────────────────

    def count_active_edges(self) -> int:
        """Count edges that are not collapsed."""
        return sum(1 for e in self.edges.values() if not e.blocked)

    def count_blocked_edges(self) -> int:
        """Count edges that are collapsed."""
        return sum(1 for e in self.edges.values() if e.blocked)

    def find_disconnected_nodes(self) -> list:
        """Find sectors with zero active neighbors (isolated sectors)."""
        return [nid for nid in self.nodes if not self.get_neighbors(nid)]

    # ── Serialization ───────────────────────────────────────

    def to_dict(self) -> dict:
        """
        Serialize the entire grid state for API / frontend.
        Ready for JSON serialization via FastAPI or WebSocket.
        """
        return {
            "grid_rows": QuakeConfig.GRID_ROWS,
            "grid_cols": QuakeConfig.GRID_COLS,
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "edges": [edge.to_dict() for edge in self.edges.values()],
            "active_edges": self.count_active_edges(),
            "blocked_edges": self.count_blocked_edges(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DisasterGrid":
        """
        Reconstruct a DisasterGrid from serialized data.
        Useful for loading saved states or receiving from API.
        """
        graph = cls()
        # Restore node attributes
        for node_data in data.get("nodes", []):
            nid = node_data["id"]
            if nid in graph.nodes:
                node = graph.nodes[nid]
                node.node_type = node_data.get("node_type", QuakeConfig.EMPTY)
                node.population = node_data.get("population", 0)
                node.vulnerability_index = node_data.get("vulnerability_index", 0.0)
                node.collapse_risk = node_data.get("collapse_risk", "NONE")
                node.accessible = node_data.get("accessible", True)
        # Restore edges
        for edge_data in data.get("edges", []):
            a, b = edge_data["node_a"], edge_data["node_b"]
            graph.add_edge(a, b, edge_data.get("cost", QuakeConfig.COST_STANDARD))
            if edge_data.get("blocked", False):
                graph.collapse_road(a, b)
        return graph
