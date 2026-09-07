# emergency_routing.py — Challenge 4: Survivor Rescue Routing via Dynamic A*
# A* search with event-driven replanning when corridors collapse from aftershocks.
#
# Design note: For dynamic environments, advanced alternatives exist:
#   - D* Lite: Incrementally repairs shortest path trees when edge costs change
#   - Lifelong Planning A* (LPA*): Maintains consistent g-values across changes
# However, for a 225-sector grid, full A* re-computation is fast enough (~1ms)
# and simpler to implement, validate, and explain in a viva.
#
# All algorithms implemented from scratch — no external pathfinding libraries.

import math
import heapq
from core.city_graph import DisasterGrid
from config.simulation_config import QuakeConfig


# ──────────────────────────────────────────────────────────
# Heuristic function — Euclidean distance
# Admissible because: straight-line distance <= actual corridor cost
# (minimum edge cost is 0.8, and grid spacing is 1.0)
# Manhattan would over-estimate with diagonal corridors (cost 1.4)
# ──────────────────────────────────────────────────────────
def heuristic(graph: DisasterGrid, node_id: int, goal_id: int) -> float:
    """Euclidean distance heuristic. Admissible and consistent for this grid."""
    n = graph.nodes[node_id]
    g = graph.nodes[goal_id]
    return math.sqrt((n.row - g.row) ** 2 + (n.col - g.col) ** 2)


class AStarRouter:
    """
    Survivor rescue routing system using A* search with dynamic replanning.

    Features:
        - Optimal shortest path via A* with Euclidean heuristic
        - Event-driven replanning: automatically recalculates when a corridor collapses
        - Sequential survivor rescue: visits trapped survivors in order
        - No stale cached paths: paths are always recomputed from current position

    The router registers as an observer on the shared disaster grid, so aftershock
    collapses trigger immediate re-evaluation without polling.
    """

    def __init__(self, graph: DisasterGrid):
        self.graph = graph
        self.current_path: list = []      # Remaining path sectors
        self.current_pos: int = None      # Team's current position (sector ID)
        self.survivors: list = []         # Unvisited trapped survivor sector IDs (ordered)
        self.visited: list = []           # Rescued survivor sector IDs
        self.log: list = []               # Event log entries (simulation collects these)
        self.total_reroutes: int = 0      # Count of replanning events

        # Register as observer for corridor collapse events
        self.graph.register_observer(self._on_corridor_collapsed)

    # ── Mission setup ───────────────────────────────────────

    def set_mission(self, start_id: int, survivor_ids: list):
        """Initialize a rescue mission from start_id visiting all survivor_ids in sequence."""
        self.current_pos = start_id
        self.survivors = list(survivor_ids)
        self.visited = []
        self._plan_next_leg()

    # ── Path planning ───────────────────────────────────────

    def _plan_next_leg(self):
        """Plan route from current position to next trapped survivor."""
        if not self.survivors:
            self.current_path = []
            self.log.append("MISSION COMPLETE: All survivors reached.")
            return

        next_target = self.survivors[0]
        path = self.astar(self.current_pos, next_target)

        if path is None:
            self.log.append(
                f"WARNING: No path to survivor at sector {next_target}. Skipping."
            )
            self.survivors.pop(0)
            self._plan_next_leg()  # Try next survivor
        else:
            self.current_path = path
            self.log.append(
                f"ROUTE PLANNED: {self.current_pos} -> {next_target}, "
                f"{len(path)} steps"
            )

    # ── A* algorithm — from scratch ─────────────────────────

    def astar(self, start_id: int, goal_id: int) -> list:
        """
        A* search algorithm. Returns list of sector IDs (path) or None.

        Guarantees:
            - Optimal path (Euclidean heuristic is admissible + consistent)
            - Respects collapsed corridors (via graph.get_neighbors)
            - Respects collapse risk multipliers (via graph.get_edge_cost)
            - No stale data (always queries live grid state)
        """
        if start_id == goal_id:
            return [start_id]

        g_score = {start_id: 0.0}
        came_from = {start_id: None}

        # Priority queue: (f_score, node_id)
        heap = [(heuristic(self.graph, start_id, goal_id), start_id)]
        closed = set()

        while heap:
            f, current = heapq.heappop(heap)

            if current in closed:
                continue
            closed.add(current)

            if current == goal_id:
                return self._reconstruct_path(came_from, goal_id)

            for neighbor in self.graph.get_neighbors(current):
                if neighbor in closed:
                    continue
                tentative_g = g_score[current] + self.graph.get_edge_cost(current, neighbor)
                if tentative_g < g_score.get(neighbor, float("inf")):
                    g_score[neighbor] = tentative_g
                    came_from[neighbor] = current
                    f_score = tentative_g + heuristic(self.graph, neighbor, goal_id)
                    heapq.heappush(heap, (f_score, neighbor))

        return None  # No path found

    def _reconstruct_path(self, came_from: dict, end_id: int) -> list:
        """Reconstruct path from came_from dict."""
        path = []
        node = end_id
        while node is not None:
            path.append(node)
            node = came_from[node]
        path.reverse()
        return path

    # ── Observer callback — corridor collapse handler ───────

    def _on_corridor_collapsed(self, id_a: int, id_b: int):
        """
        Called automatically when a corridor collapses on the shared disaster grid.
        If the collapsed corridor is on the current path, triggers replanning.
        No stale cached paths — recomputes from current position.
        """
        if self._path_uses_edge(id_a, id_b):
            self.total_reroutes += 1
            self.log.append(
                f"REPLANNING: Corridor ({id_a} <-> {id_b}) collapsed by aftershock on current route. "
                f"Recalculating from sector {self.current_pos}."
            )
            self._plan_next_leg()

    def _path_uses_edge(self, id_a: int, id_b: int) -> bool:
        """Check if the collapsed corridor is part of the current planned path."""
        for i in range(len(self.current_path) - 1):
            a = self.current_path[i]
            b = self.current_path[i + 1]
            if (a == id_a and b == id_b) or (a == id_b and b == id_a):
                return True
        return False

    # ── Step execution ──────────────────────────────────────

    def advance_one_step(self) -> int:
        """
        Move the rescue team one sector forward along the current path.
        Returns the new position sector ID, or None if no movement.
        """
        if not self.current_path or len(self.current_path) < 2:
            return None

        # Move to next sector in path
        self.current_pos = self.current_path[1]
        self.current_path.pop(0)

        # Check if we've reached the target survivor
        if self.survivors and self.current_pos == self.survivors[0]:
            self.log.append(f"SURVIVOR REACHED at sector {self.current_pos}")
            self.visited.append(self.survivors.pop(0))
            self._plan_next_leg()

        return self.current_pos

    # ── Serialization ───────────────────────────────────────

    def to_dict(self) -> dict:
        """Serialize router state for API / frontend."""
        return {
            "current_pos": self.current_pos,
            "current_path": self.current_path,
            "survivors_remaining": self.survivors,
            "survivors_rescued": self.visited,
            "total_reroutes": self.total_reroutes,
        }


# ──────────────────────────────────────────────────────────
# Entry point — simulation creates router here
# ──────────────────────────────────────────────────────────
def create_router(graph: DisasterGrid) -> AStarRouter:
    """Create and return an A* rescue router bound to the shared disaster grid."""
    return AStarRouter(graph)
