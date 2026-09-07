# layout_planning.py — Challenge 1: Emergency Resource Zone Planning via CSP
# Uses Backtracking + MRV heuristic + Forward Checking to assign zone types.
#
# Strategy: Pure backtracking on 225 sectors is too slow.
# We use a hybrid approach:
#   1. Seed required special types greedy with MRV ordering (most constrained first)
#   2. Fill remaining sectors with RESIDENTIAL/EMPTY
#   3. Repair C2/C3 hop constraint violations via BFS
#   4. Final validation of all constraints

import random
from collections import Counter, deque
from core.city_graph import DisasterGrid
from config.simulation_config import QuakeConfig


class CSPSolver:
    """
    Constraint Satisfaction Problem solver for emergency resource zone planning.

    After an earthquake, the city must be divided into functional zones:
        - Field Hospitals for treating injured survivors
        - Shelters for displaced residents
        - Hazard Zones (collapsed industrial areas — too dangerous for shelter/hospital)
        - Generator Stations for emergency power
        - Supply Depots for distributing relief supplies

    Constraints:
        C1: Hazard Zones cannot be adjacent to Shelters or Field Hospitals
            (structural collapse risk endangers survivors)
        C2: Every Residential block must have a Field Hospital within 3 blocks
            (maximum triage response distance)
        C3: Every Generator Station must have a Hazard Zone within 2 blocks
            (power routing to critical industrial infrastructure)
        C4: If no valid layout exists, identify which constraint failed

    Uses MRV (Minimum Remaining Values) for variable selection
    and Forward Checking for constraint propagation.
    """

    def __init__(self, graph: DisasterGrid):
        self.graph = graph
        self.assignment: dict = {}            # node_id -> node_type
        self.domains: dict = {}               # node_id -> list of possible types
        self.remaining_counts: dict = dict(QuakeConfig.REQUIRED_COUNTS)

    def solve(self) -> bool:
        """
        Main entry point. Attempts to find a valid emergency resource zone layout.
        Returns True if a consistent layout was found.
        """
        # Initialize domains — each sector can be any type initially
        for nid in self.graph.nodes:
            self.domains[nid] = list(QuakeConfig.ALL_TYPES)

        # Phase 1: Seed special types using greedy placement with MRV ordering
        if not self._seed_special_types():
            return False

        # Phase 2: Fill remaining unassigned sectors with RESIDENTIAL / EMPTY
        self._fill_remaining()

        # Phase 3: Repair hop constraint violations (C2, C3)
        self._repair_hop_constraints()

        # Phase 4: Validate all constraints
        return self._is_complete_assignment_valid()

    # ── Phase 1: Greedy seeding of special types ────────────

    def _seed_special_types(self) -> bool:
        """Place required special types in MRV order. Hazard Zones first (most constrained by C1)."""
        ordered = [
            QuakeConfig.HAZARD_ZONE, QuakeConfig.FIELD_HOSPITAL, QuakeConfig.SHELTER,
            QuakeConfig.GENERATOR_STATION, QuakeConfig.SUPPLY_DEPOT,
        ]
        for node_type in ordered:
            needed = self.remaining_counts.get(node_type, 0)
            placed = self._place_type_greedy(node_type, needed)
            if placed < needed:
                # Retry with relaxed selection
                placed += self._place_type_greedy(node_type, needed - placed)
            if placed < needed:
                return False
        return True

    def _place_type_greedy(self, node_type: str, count: int) -> int:
        """Place up to 'count' sectors of the given type using MRV ordering."""
        candidates = [
            nid for nid in self.graph.nodes
            if nid not in self.assignment and node_type in self.domains[nid]
        ]
        random.shuffle(candidates)
        # MRV: smallest domain first (most constrained)
        candidates.sort(key=lambda nid: len(self.domains[nid]))

        placed = 0
        for nid in candidates:
            if placed >= count:
                break
            if self._is_consistent(nid, node_type):
                self._assign(nid, node_type)
                placed += 1
        return placed

    # ── Phase 2: Fill remaining sectors ─────────────────────

    def _fill_remaining(self):
        """Fill unassigned sectors with RESIDENTIAL (up to required count) then EMPTY."""
        res_needed = max(0, self.remaining_counts.get(QuakeConfig.RESIDENTIAL, 0))
        unassigned = [nid for nid in self.graph.nodes if nid not in self.assignment]
        random.shuffle(unassigned)
        placed_res = 0
        for nid in unassigned:
            if placed_res < res_needed and QuakeConfig.RESIDENTIAL in self.domains[nid]:
                self._assign(nid, QuakeConfig.RESIDENTIAL)
                placed_res += 1
            else:
                self._assign(nid, QuakeConfig.EMPTY)

    # ── Phase 3: Repair hop constraint violations ───────────

    def _repair_hop_constraints(self):
        """Fix C2 and C3 violations by adding missing field hospitals/hazard zones nearby."""
        hospitals = [n.id for n in self.graph.get_nodes_of_type(QuakeConfig.FIELD_HOSPITAL)]
        hazard_zones = [n.id for n in self.graph.get_nodes_of_type(QuakeConfig.HAZARD_ZONE)]

        # C2: Residential sectors too far from field hospital
        for node in list(self.graph.get_nodes_of_type(QuakeConfig.RESIDENTIAL)):
            if not hospitals:
                break
            min_hops = min(self._bfs_grid_hops(node.id, h) for h in hospitals)
            if min_hops < 0 or min_hops > QuakeConfig.HOSPITAL_HOP_LIMIT:
                self._try_add_type_near(node.id, QuakeConfig.FIELD_HOSPITAL, hospitals,
                                        QuakeConfig.HOSPITAL_HOP_LIMIT)

        # C3: Generator stations too far from hazard zone
        for node in list(self.graph.get_nodes_of_type(QuakeConfig.GENERATOR_STATION)):
            if not hazard_zones:
                break
            min_hops = min(self._bfs_grid_hops(node.id, i) for i in hazard_zones)
            if min_hops < 0 or min_hops > QuakeConfig.HAZARD_HOP_LIMIT:
                self._try_add_type_near(node.id, QuakeConfig.HAZARD_ZONE, hazard_zones,
                                        QuakeConfig.HAZARD_HOP_LIMIT)

    def _try_add_type_near(self, source_id: int, target_type: str,
                           existing_list: list, max_hops: int):
        """BFS outward from source_id and convert an EMPTY sector to target_type."""
        visited = {source_id}
        queue = deque([(source_id, 0)])
        while queue:
            cur, dist = queue.popleft()
            if dist > max_hops:
                break
            cur_node = self.graph.nodes[cur]
            if cur != source_id and cur_node.node_type == QuakeConfig.EMPTY:
                if self._is_consistent(cur, target_type):
                    cur_node.node_type = target_type
                    self.assignment[cur] = target_type
                    existing_list.append(cur)
                    return
            for nb in self.graph.get_grid_neighbors(cur):
                if nb not in visited:
                    visited.add(nb)
                    queue.append((nb, dist + 1))

    # ── MRV Variable Selection (exposed for viva) ──────────

    def select_unassigned_variable(self) -> int:
        """
        MRV (Minimum Remaining Values) heuristic.
        Selects the unassigned variable with the smallest domain.
        Tie-break: highest degree (most unassigned neighbors).
        """
        unassigned = [nid for nid in self.graph.nodes if nid not in self.assignment]
        if not unassigned:
            return None
        min_domain = min(len(self.domains[nid]) for nid in unassigned)
        candidates = [nid for nid in unassigned if len(self.domains[nid]) == min_domain]
        if len(candidates) > 1:
            def degree(nid):
                return sum(1 for nb in self.graph.get_grid_neighbors(nid)
                           if nb not in self.assignment)
            candidates.sort(key=degree, reverse=True)
        return candidates[0]

    # ── Core CSP operations ─────────────────────────────────

    def _assign(self, node_id: int, value: str):
        """Assign a type to a sector and propagate constraints via forward checking."""
        self.assignment[node_id] = value
        self.graph.nodes[node_id].node_type = value
        self.domains[node_id] = [value]
        if value in self.remaining_counts and self.remaining_counts[value] > 0:
            self.remaining_counts[value] -= 1
        self._forward_check(node_id, value)

    def _is_consistent(self, node_id: int, value: str) -> bool:
        """Check if assigning 'value' to node_id violates C1 with any assigned neighbor."""
        for nb_id in self.graph.get_grid_neighbors(node_id):
            nb_type = self.assignment.get(nb_id)
            if nb_type and not self._values_compatible(value, nb_type):
                return False
        return True

    def _values_compatible(self, val_a: str, val_b: str) -> bool:
        """Check C1: Hazard Zone cannot be adjacent to Field Hospital or Shelter."""
        if val_a == QuakeConfig.HAZARD_ZONE and val_b in (QuakeConfig.FIELD_HOSPITAL, QuakeConfig.SHELTER):
            return False
        if val_b == QuakeConfig.HAZARD_ZONE and val_a in (QuakeConfig.FIELD_HOSPITAL, QuakeConfig.SHELTER):
            return False
        return True

    def _forward_check(self, node_id: int, value: str) -> bool:
        """
        Forward checking: prune incompatible values from neighbors' domains.
        After assigning 'value' to node_id, remove any neighbor domain value
        that would violate C1.
        """
        for nb in self.graph.get_grid_neighbors(node_id):
            if nb in self.assignment:
                continue
            pruned = [v for v in self.domains[nb] if self._values_compatible(value, v)]
            self.domains[nb] = pruned if pruned else [QuakeConfig.EMPTY]
        return True

    # ── BFS hop distance (grid-based, no edges needed) ──────

    def _bfs_grid_hops(self, start_id: int, end_id: int) -> int:
        """BFS on grid neighbors (not road edges). Returns hop count or -1."""
        if start_id == end_id:
            return 0
        visited = {start_id}
        queue = deque([(start_id, 0)])
        while queue:
            cur, dist = queue.popleft()
            for nb in self.graph.get_grid_neighbors(cur):
                if nb == end_id:
                    return dist + 1
                if nb not in visited:
                    visited.add(nb)
                    queue.append((nb, dist + 1))
        return -1

    # ── Final validation ────────────────────────────────────

    def _is_complete_assignment_valid(self) -> bool:
        """Validate all constraints on the completed assignment."""
        # C1: Hazard Zone not adjacent to Field Hospital/Shelter
        for nid, node in self.graph.nodes.items():
            if node.node_type == QuakeConfig.HAZARD_ZONE:
                for nb_id in self.graph.get_grid_neighbors(nid):
                    if self.graph.nodes[nb_id].node_type in (QuakeConfig.FIELD_HOSPITAL, QuakeConfig.SHELTER):
                        return False

        hospitals = [n.id for n in self.graph.get_nodes_of_type(QuakeConfig.FIELD_HOSPITAL)]
        hazard_zones = [n.id for n in self.graph.get_nodes_of_type(QuakeConfig.HAZARD_ZONE)]
        if not hospitals or not hazard_zones:
            return False

        # C2: Every Residential within 3 blocks of a Field Hospital
        for node in self.graph.get_nodes_of_type(QuakeConfig.RESIDENTIAL):
            min_hops = min(self._bfs_grid_hops(node.id, h) for h in hospitals)
            if min_hops < 0 or min_hops > QuakeConfig.HOSPITAL_HOP_LIMIT:
                return False

        # C3: Every Generator Station within 2 blocks of a Hazard Zone
        for node in self.graph.get_nodes_of_type(QuakeConfig.GENERATOR_STATION):
            min_hops = min(self._bfs_grid_hops(node.id, i) for i in hazard_zones)
            if min_hops < 0 or min_hops > QuakeConfig.HAZARD_HOP_LIMIT:
                return False

        # Count validation
        counts = Counter(n.node_type for n in self.graph.nodes.values())
        for t, required in QuakeConfig.REQUIRED_COUNTS.items():
            if counts.get(t, 0) < required:
                return False
        return True

    # ── Post-solve: apply to graph ──────────────────────────

    def apply_assignment_to_graph(self):
        """Sync assignment dict with graph and populate sector attributes."""
        for nid, node in self.graph.nodes.items():
            self.assignment[nid] = node.node_type
        self.graph.populate_after_layout()

    # ── Conflict identification ─────────────────────────────

    def identify_failing_constraint(self) -> str:
        """
        Identify which constraint caused the layout to fail.
        Returns a human-readable conflict description.
        """
        counts = Counter(n.node_type for n in self.graph.nodes.values())

        if counts.get(QuakeConfig.FIELD_HOSPITAL, 0) < QuakeConfig.REQUIRED_COUNTS.get(QuakeConfig.FIELD_HOSPITAL, 0):
            return (f"CONFLICT: Only {counts.get(QuakeConfig.FIELD_HOSPITAL, 0)} field hospitals placed, "
                    f"{QuakeConfig.REQUIRED_COUNTS[QuakeConfig.FIELD_HOSPITAL]} required. "
                    f"C1 adjacency constraint blocked field hospital placement near hazard zones.")

        if counts.get(QuakeConfig.HAZARD_ZONE, 0) < QuakeConfig.REQUIRED_COUNTS.get(QuakeConfig.HAZARD_ZONE, 0):
            return (f"CONFLICT: Only {counts.get(QuakeConfig.HAZARD_ZONE, 0)} hazard zones placed. "
                    f"C3 generator-station proximity constraint cannot be satisfied.")

        # Check C2 violations
        hospitals = [n.id for n in self.graph.get_nodes_of_type(QuakeConfig.FIELD_HOSPITAL)]
        if hospitals:
            for node in self.graph.get_nodes_of_type(QuakeConfig.RESIDENTIAL):
                min_hops = min(self._bfs_grid_hops(node.id, h) for h in hospitals)
                if min_hops < 0 or min_hops > QuakeConfig.HOSPITAL_HOP_LIMIT:
                    return (f"CONFLICT: Residential sector ({node.row},{node.col}) is "
                            f"{min_hops} blocks from nearest field hospital. "
                            f"C2 requires within {QuakeConfig.HOSPITAL_HOP_LIMIT} blocks. "
                            f"Suggested fix: add field hospital near ({node.row},{node.col}).")

        return "CONFLICT: C2 (Residential-Field Hospital 3-block rule) cannot be satisfied."

    def suggest_minimum_conflict_adjustment(self) -> list:
        """
        Propose minimum changes to fix constraint violations.
        Returns list of (node_id, current_type, suggested_type, reason) tuples.
        """
        suggestions = []
        hospitals = [n.id for n in self.graph.get_nodes_of_type(QuakeConfig.FIELD_HOSPITAL)]

        # Find residential sectors violating C2
        for node in self.graph.get_nodes_of_type(QuakeConfig.RESIDENTIAL):
            if not hospitals:
                break
            min_hops = min(self._bfs_grid_hops(node.id, h) for h in hospitals)
            if min_hops < 0 or min_hops > QuakeConfig.HOSPITAL_HOP_LIMIT:
                # Find nearest EMPTY sector to convert
                for nb in self.graph.get_grid_neighbors(node.id):
                    if self.graph.nodes[nb].node_type == QuakeConfig.EMPTY:
                        suggestions.append((
                            nb, QuakeConfig.EMPTY, QuakeConfig.FIELD_HOSPITAL,
                            f"Convert to field hospital: residential sector ({node.row},{node.col}) "
                            f"needs field hospital within {QuakeConfig.HOSPITAL_HOP_LIMIT} blocks"
                        ))
                        break
        return suggestions
