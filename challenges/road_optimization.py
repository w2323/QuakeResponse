# road_optimization.py — Challenge 2: Emergency Corridor Network Optimization
# Kruskal's MST + redundancy corridors + edge-disjoint hospital-depot lifeline guarantee.
#
# Approach (per instructor correction):
#   1. Build MST with Kruskal's algorithm (minimum spanning tree of all 225 sectors)
#   2. Add extra redundancy corridors to eliminate dead ends
#   3. Verify and enforce 2 edge-disjoint paths between Primary Field Hospital and Supply Depot
#
# All algorithms implemented from scratch — no networkx.

import random
from collections import deque
from core.city_graph import DisasterGrid
from config.simulation_config import QuakeConfig


# ──────────────────────────────────────────────────────────
# Generate candidate corridors (4-directional + diagonal)
# ──────────────────────────────────────────────────────────
def generate_candidate_edges(graph: DisasterGrid) -> list:
    """
    Generate all possible grid-adjacent corridors as (cost, id_a, id_b) tuples.
    4-directional: cost 1.0 (or 0.8 if both endpoints are RESIDENTIAL)
    Diagonal: cost 1.4
    """
    edges = []
    for row in range(QuakeConfig.GRID_ROWS):
        for col in range(QuakeConfig.GRID_COLS):
            node = graph.get_node(row, col)

            # Right neighbor
            if col + 1 < QuakeConfig.GRID_COLS:
                nb = graph.get_node(row, col + 1)
                cost = (QuakeConfig.COST_RESIDENTIAL
                        if node.node_type == QuakeConfig.RESIDENTIAL
                        and nb.node_type == QuakeConfig.RESIDENTIAL
                        else QuakeConfig.COST_STANDARD)
                edges.append((cost, node.id, nb.id))

            # Down neighbor
            if row + 1 < QuakeConfig.GRID_ROWS:
                nb = graph.get_node(row + 1, col)
                cost = (QuakeConfig.COST_RESIDENTIAL
                        if node.node_type == QuakeConfig.RESIDENTIAL
                        and nb.node_type == QuakeConfig.RESIDENTIAL
                        else QuakeConfig.COST_STANDARD)
                edges.append((cost, node.id, nb.id))

            # Diagonal down-right
            if row + 1 < QuakeConfig.GRID_ROWS and col + 1 < QuakeConfig.GRID_COLS:
                nb = graph.get_node(row + 1, col + 1)
                edges.append((QuakeConfig.COST_DIAGONAL, node.id, nb.id))

            # Diagonal down-left
            if row + 1 < QuakeConfig.GRID_ROWS and col - 1 >= 0:
                nb = graph.get_node(row + 1, col - 1)
                edges.append((QuakeConfig.COST_DIAGONAL, node.id, nb.id))

    return edges


# ──────────────────────────────────────────────────────────
# Union-Find — implemented from scratch for Kruskal's
# ──────────────────────────────────────────────────────────
class UnionFind:
    """Union-Find (Disjoint Set) with path compression and union by rank."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        """Find root with path compression."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        """Union by rank. Returns True if x and y were in different sets."""
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True


# ──────────────────────────────────────────────────────────
# Step 1: Kruskal's MST — from scratch
# ──────────────────────────────────────────────────────────
def build_mst(graph: DisasterGrid) -> list:
    """
    Build Minimum Spanning Tree using Kruskal's algorithm.
    Returns list of (id_a, id_b, cost) for MST corridors.
    Also adds those corridors to the shared disaster grid.
    """
    candidates = generate_candidate_edges(graph)
    candidates.sort()  # Sort by cost (greedy choice)
    uf = UnionFind(QuakeConfig.GRID_ROWS * QuakeConfig.GRID_COLS)
    mst_edges = []

    for cost, a, b in candidates:
        if uf.union(a, b):
            mst_edges.append((a, b, cost))
            graph.add_edge(a, b, cost)
            if len(mst_edges) == QuakeConfig.GRID_ROWS * QuakeConfig.GRID_COLS - 1:
                break  # MST complete (V-1 edges)

    return mst_edges


# ──────────────────────────────────────────────────────────
# Step 2: Add redundancy corridors — eliminate dead ends
# ──────────────────────────────────────────────────────────
def add_redundancy_edges(graph: DisasterGrid, extra_count: int = None):
    """
    Add extra corridors beyond MST to improve network resilience against aftershocks.
    Priority: dead-end sectors (degree == 1) first, then cheapest available.
    """
    if extra_count is None:
        extra_count = QuakeConfig.REDUNDANCY_EDGES

    candidates = generate_candidate_edges(graph)
    existing = set(graph.edges.keys())

    # Only corridors not already in the grid
    non_mst = [(c, a, b) for c, a, b in candidates
               if (min(a, b), max(a, b)) not in existing]
    non_mst.sort()

    # Dead-end sectors — only 1 active neighbor
    dead_ends = {nid for nid in graph.nodes if len(graph.adj[nid]) == 1}

    added = 0

    # Priority 1: Connect dead-end sectors
    for cost, a, b in non_mst:
        if added >= extra_count:
            break
        if a in dead_ends or b in dead_ends:
            graph.add_edge(a, b, cost)
            added += 1

    # Priority 2: Cheapest remaining corridors
    for cost, a, b in non_mst:
        if added >= extra_count:
            break
        key = (min(a, b), max(a, b))
        if key not in graph.edges:
            graph.add_edge(a, b, cost)
            added += 1


# ──────────────────────────────────────────────────────────
# BFS path helpers — for edge-disjoint path verification
# ──────────────────────────────────────────────────────────
def bfs_path(graph: DisasterGrid, start: int, end: int) -> list:
    """Standard BFS. Returns list of node IDs from start to end, or None."""
    if start == end:
        return [start]
    visited = {start: None}
    queue = deque([start])
    while queue:
        cur = queue.popleft()
        for nb in graph.get_neighbors(cur):
            if nb not in visited:
                visited[nb] = cur
                if nb == end:
                    path = []
                    node = end
                    while node is not None:
                        path.append(node)
                        node = visited[node]
                    path.reverse()
                    return path
                queue.append(nb)
    return None


def bfs_path_excluding(graph: DisasterGrid, start: int, end: int,
                       excluded_edges: set) -> list:
    """BFS that skips corridors whose canonical key is in excluded_edges."""
    if start == end:
        return [start]
    visited = {start: None}
    queue = deque([start])
    while queue:
        cur = queue.popleft()
        for nb in graph.get_neighbors(cur):
            key = (min(cur, nb), max(cur, nb))
            if key in excluded_edges:
                continue
            if nb not in visited:
                visited[nb] = cur
                if nb == end:
                    path = []
                    node = end
                    while node is not None:
                        path.append(node)
                        node = visited[node]
                    path.reverse()
                    return path
                queue.append(nb)
    return None


# ──────────────────────────────────────────────────────────
# Step 3: Ensure 2 edge-disjoint paths between Field Hospital and Supply Depot
#
# This uses the concept from max-flow/min-cut:
# Find path P1, then find path P2 that shares NO edges with P1.
# If P2 doesn't exist, incrementally add corridors until it does.
# This guarantees true edge-disjoint redundancy (not just
# different shortest paths that might share critical edges).
# ──────────────────────────────────────────────────────────
def ensure_hospital_depot_lifeline(graph: DisasterGrid) -> bool:
    """
    Verify and enforce 2 edge-disjoint paths between Primary Field Hospital
    and Primary Supply Depot.

    Edge-disjoint means: no single aftershock can disconnect both paths.
    This is stronger than just "two different paths" — it uses the
    max-flow/min-cut principle (if min-cut >= 2, then 2 edge-disjoint
    paths exist by Menger's theorem).

    Note: For production systems, D* Lite or Lifelong Planning A* could
    provide more efficient dynamic replanning, but for this 225-sector grid,
    BFS-based verification is sufficient and simpler to validate.
    """
    hospitals = graph.get_nodes_of_type(QuakeConfig.FIELD_HOSPITAL)
    depots = graph.get_nodes_of_type(QuakeConfig.SUPPLY_DEPOT)
    if not hospitals or not depots:
        return False

    h_id = hospitals[0].id    # Primary field hospital
    d_id = depots[0].id       # Primary supply depot

    # Find first path via BFS
    path1 = bfs_path(graph, h_id, d_id)
    if path1 is None:
        return False

    # Collect corridors used by path1
    used_edges = set()
    for i in range(len(path1) - 1):
        used_edges.add((min(path1[i], path1[i + 1]), max(path1[i], path1[i + 1])))

    # Try to find second path that shares NO corridors with path1
    path2 = bfs_path_excluding(graph, h_id, d_id, used_edges)
    if path2 is not None:
        return True  # Already have 2 edge-disjoint paths

    # No second path found — incrementally add corridors until one exists
    candidates = generate_candidate_edges(graph)
    candidates.sort()
    for cost, a, b in candidates:
        key = (min(a, b), max(a, b))
        if key not in graph.edges and key not in used_edges:
            graph.add_edge(a, b, cost)
            path2 = bfs_path_excluding(graph, h_id, d_id, used_edges)
            if path2 is not None:
                return True

    return False


def find_bridges(graph: DisasterGrid) -> list:
    """
    Find bridge corridors (corridors whose collapse disconnects the grid).
    Uses Tarjan's bridge-finding algorithm.
    Useful for identifying critical single-points-of-failure in the corridor network.
    """
    disc = {}
    low = {}
    parent = {}
    bridges = []
    timer = [0]

    def dfs(u):
        disc[u] = low[u] = timer[0]
        timer[0] += 1
        for v in graph.get_neighbors(u):
            if v not in disc:
                parent[v] = u
                dfs(v)
                low[u] = min(low[u], low[v])
                if low[v] > disc[u]:
                    bridges.append((min(u, v), max(u, v)))
            elif v != parent.get(u):
                low[u] = min(low[u], disc[v])

    # Run from all unvisited sectors (handles disconnected components)
    for nid in graph.nodes:
        if nid not in disc:
            parent[nid] = -1
            dfs(nid)

    return bridges


# ──────────────────────────────────────────────────────────
# Main entry point — simulation calls this
# ──────────────────────────────────────────────────────────
def build_emergency_corridors(graph: DisasterGrid) -> bool:
    """
    Build the complete emergency corridor network:
    1. Kruskal's MST for minimum-cost spanning tree
    2. Redundancy corridors to eliminate dead ends
    3. Edge-disjoint hospital-depot lifeline guarantee
    """
    build_mst(graph)
    add_redundancy_edges(graph)
    success = ensure_hospital_depot_lifeline(graph)
    if not success:
        raise RuntimeError(
            "Critical: Could not establish 2 edge-disjoint lifeline paths "
            "between Primary Field Hospital and Supply Depot."
        )
    return True
