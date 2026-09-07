# simulation_engine.py — QuakeResponse Disaster Simulation Engine
# Contains SimulationState (single source of truth) and Simulation runner.
# No GUI dependencies. Fully terminal-runnable.

import random
import time
from core.city_graph import DisasterGrid
from core.event_manager import EventManager, EventType
from config.simulation_config import QuakeConfig


# ──────────────────────────────────────────────────────────
# SimulationState — Single Source of Truth
# This is what gets serialized for API/frontend consumption.
# ──────────────────────────────────────────────────────────
class SimulationState:
    """
    Centralized simulation state. Contains everything needed to
    describe the current state of the disaster response simulation.

    This is the SINGLE SOURCE OF TRUTH — not just DisasterGrid.
    All modules read from and write to this state object.

    Designed for future serialization to:
        - FastAPI endpoints
        - WebSocket real-time updates
        - React frontend state
        - Replay systems
    """

    def __init__(self, graph: DisasterGrid, events: EventManager):
        self.graph: DisasterGrid = graph
        self.events: EventManager = events
        self.step: int = 0
        self.max_steps: int = QuakeConfig.MAX_STEPS
        self.medical_units: list = []        # Current medical unit node IDs
        self.survivors: list = []            # Active trapped survivor node IDs
        self.aftershock_history: list = []   # (step, id_a, id_b) tuples
        self.risk_predictions: dict = {}     # node_id -> collapse_risk_level
        self.active_routes: list = []        # Current planned route path
        self.sar_deployments: list = []      # [(node_id, risk_level, score)] for SAR teams
        self.router = None                   # AStarRouter instance
        self.is_initialized: bool = False
        self.is_complete: bool = False

    def to_dict(self) -> dict:
        """
        Serialize the complete simulation state for API consumption.
        This is what a React frontend would receive via WebSocket.
        """
        result = {
            "step": self.step,
            "max_steps": self.max_steps,
            "is_initialized": self.is_initialized,
            "is_complete": self.is_complete,
            "graph": self.graph.to_dict(),
            "medical_units": self.medical_units,
            "survivors_remaining": self.survivors,
            "aftershock_history": [
                {"step": s, "node_a": a, "node_b": b}
                for s, a, b in self.aftershock_history
            ],
            "risk_predictions": self.risk_predictions,
            "active_routes": self.active_routes,
            "sar_deployments": [
                {"node_id": nid, "collapse_risk": rl, "priority_score": sc}
                for nid, rl, sc in self.sar_deployments
            ],
            "events": self.events.to_dict(),
        }
        if self.router:
            result["router"] = self.router.to_dict()
        return result


# ──────────────────────────────────────────────────────────
# Simulation — the main disaster response simulation runner
# ──────────────────────────────────────────────────────────
class Simulation:
    """
    20-step disaster response simulation engine that integrates all 5 challenges.

    Initialization order (dependency-driven):
        1. CSP (resource zones) → assigns zone types to sectors
        2. MST (corridors) → builds emergency corridor network
        3. ML (vulnerability) → predicts structural collapse risk, updates edge costs
        4. GA (medical units) → deploys medical units using risk-aware costs
        5. A* (routing) → plans survivor rescue mission

    Each simulation step:
        - Random aftershock events (30% chance)
        - Rescue team advancement toward trapped survivors
        - Periodic medical unit repositioning (every 5 steps)
        - Periodic vulnerability re-evaluation (every 5 steps)
        - Comprehensive logging
    """

    def __init__(self, seed: int = None, verbose: bool = True):
        # Set deterministic seed if provided
        self._seed = seed or QuakeConfig.SEED
        if self._seed is not None:
            random.seed(self._seed)

        self.events = EventManager(verbose=verbose)
        self.graph = DisasterGrid()
        self.state = SimulationState(self.graph, self.events)

    # ── Initialization — all 5 challenges in order ──────────

    def initialize(self):
        """
        Initialize all 5 challenge modules in dependency order.
        Raises RuntimeError if any critical step fails.
        """
        self.events.metrics.simulation_start_time = time.time()
        self.events.log(EventType.SIMULATION, "Initializing QuakeResponse disaster simulation...")

        # 1. Challenge 1: Emergency resource zone planning via CSP
        self.events.log(EventType.LAYOUT, "Running CSP solver for resource zone planning...")
        from challenges.layout_planning import CSPSolver
        solver = CSPSolver(self.graph)
        if not solver.solve():
            conflict = solver.identify_failing_constraint()
            suggestions = solver.suggest_minimum_conflict_adjustment()
            self.events.log(EventType.LAYOUT, f"LAYOUT FAILED: {conflict}", level="error")
            for s in suggestions:
                self.events.log(EventType.LAYOUT, f"  Suggestion: {s[3]}", level="warning")
            raise RuntimeError(conflict)
        solver.apply_assignment_to_graph()
        self.events.log(EventType.LAYOUT, "Emergency resource zones assigned successfully via CSP.")

        # 2. Challenge 2: Emergency corridor network via Kruskal's MST
        self.events.log(EventType.ROADS, "Building emergency corridor network (MST + redundancy)...")
        from challenges.road_optimization import build_emergency_corridors
        build_emergency_corridors(self.graph)
        self.events.log(EventType.ROADS,
                        f"Emergency corridors built. "
                        f"{self.graph.count_active_edges()} active corridors.")

        # 3. Challenge 5: Structural vulnerability ML (run BEFORE GA so risk weights affect fitness)
        self.events.log(EventType.VULNERABILITY, "Running structural vulnerability assessment pipeline...")
        from challenges.crime_prediction import run_vulnerability_assessment
        kmeans, dt = run_vulnerability_assessment(self.graph)

        # Update risk predictions in state
        risk_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
        for nid, node in self.graph.nodes.items():
            self.state.risk_predictions[nid] = node.collapse_risk
            if node.collapse_risk in risk_counts:
                risk_counts[node.collapse_risk] += 1
        self.events.metrics.high_risk_zones = risk_counts["HIGH"]
        self.events.metrics.medium_risk_zones = risk_counts["MEDIUM"]
        self.events.metrics.low_risk_zones = risk_counts["LOW"]

        self.events.log(EventType.VULNERABILITY,
                        f"Vulnerability assessment complete. "
                        f"HIGH={risk_counts['HIGH']}, "
                        f"MEDIUM={risk_counts['MEDIUM']}, "
                        f"LOW={risk_counts['LOW']}")

        # Log Decision Tree rules for explainability
        self.events.log(EventType.VULNERABILITY,
                        f"Decision Tree rules:\n{dt.explain()}", level="debug")

        # Deploy SAR teams to highest-vulnerability sectors
        from challenges.crime_prediction import deploy_sar_teams
        self.state.sar_deployments = deploy_sar_teams(self.graph)
        team_nodes = [str(d[0]) for d in self.state.sar_deployments]
        self.events.log(EventType.VULNERABILITY,
                        f"{QuakeConfig.NUM_SAR_TEAMS} SAR teams deployed to sectors: {', '.join(team_nodes)}")

        # 4. Challenge 3: Medical unit deployment via GA
        self.events.log(EventType.MEDICAL, "Running GA for medical unit deployment...")
        from challenges.ambulance_placement import GAMedicalDeployer
        ga_deployer = GAMedicalDeployer(self.graph)
        self.state.medical_units = ga_deployer.run()
        self.events.metrics.ga_best_fitness = ga_deployer.best_fitness
        self.events.log(EventType.MEDICAL,
                        f"Medical units deployed at sectors {self.state.medical_units} "
                        f"(fitness={ga_deployer.best_fitness:.3f})")

        # 5. Challenge 4: Survivor rescue routing via A*
        self.events.log(EventType.ROUTING, "Creating A* rescue router...")
        from challenges.emergency_routing import create_router
        self.state.router = create_router(self.graph)
        self.state.survivors = self._identify_trapped_survivors(QuakeConfig.NUM_TRAPPED_SURVIVORS)
        hospitals = self.graph.get_nodes_of_type(QuakeConfig.FIELD_HOSPITAL)
        start_id = hospitals[0].id if hospitals else 0
        self.state.router.set_mission(start_id, self.state.survivors)
        self.events.log(EventType.ROUTING,
                        f"Rescue mission set. Start: sector {start_id}. "
                        f"Trapped survivors at: {self.state.survivors}")

        # Update state
        self.state.is_initialized = True
        self.events.metrics.total_edges_active = self.graph.count_active_edges()
        self.events.log(EventType.SIMULATION,
                        "Initialization complete. All 5 challenges ready.")

    # ── Identify trapped survivors from residential sectors ──

    def _identify_trapped_survivors(self, count: int) -> list:
        """Select random residential sectors as locations of trapped survivors."""
        residential = self.graph.get_nodes_of_type(QuakeConfig.RESIDENTIAL)
        sample_size = min(count, len(residential))
        return [n.id for n in random.sample(residential, sample_size)]

    # ── Step forward ────────────────────────────────────────

    def step_forward(self):
        """Execute one simulation step."""
        if self.state.step >= self.state.max_steps:
            return

        self.state.step += 1
        self.events.set_step(self.state.step)
        self.events.log(EventType.SIMULATION, f"--- Step {self.state.step} ---")

        # Random aftershock event (AFTERSHOCK_PROBABILITY chance)
        if random.random() < QuakeConfig.AFTERSHOCK_PROBABILITY:
            self._trigger_aftershock()

        # Advance rescue team one step toward trapped survivors
        if self.state.router:
            new_pos = self.state.router.advance_one_step()
            if new_pos is not None:
                self.events.log(EventType.ROUTING, f"Rescue team moved to sector {new_pos}")
            self.state.active_routes = list(self.state.router.current_path)

            # Collect router log entries
            for entry in self.state.router.log:
                if "SURVIVOR REACHED" in entry:
                    self.events.log(EventType.ROUTING, entry)
                    self.events.metrics.survivors_rescued += 1
                elif "WARNING" in entry:
                    self.events.log(EventType.ROUTING, entry, level="warning")
                    self.events.metrics.survivors_unreachable += 1
                elif "REPLANNING" in entry:
                    self.events.log(EventType.ROUTING, entry)
                    self.events.metrics.total_reroutes += 1
                elif "MISSION COMPLETE" in entry:
                    self.events.log(EventType.ROUTING, entry)
                else:
                    self.events.log(EventType.ROUTING, entry, level="debug")
            self.state.router.log.clear()

        # Periodic medical unit re-evaluation
        if self.state.step % QuakeConfig.MEDICAL_REEVAL_INTERVAL == 0:
            from challenges.ambulance_placement import redeploy_medical_units_fast
            self.state.medical_units = redeploy_medical_units_fast(self.graph)
            self.events.log(EventType.MEDICAL,
                            f"Medical units re-evaluated. Now at {self.state.medical_units}")
            self.events.metrics.medical_repositions += 1

        # Periodic vulnerability re-evaluation
        if self.state.step % QuakeConfig.COLLAPSE_REEVAL_INTERVAL == 0:
            from challenges.crime_prediction import run_vulnerability_assessment, deploy_sar_teams
            run_vulnerability_assessment(self.graph)
            risk_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
            for nid, node in self.graph.nodes.items():
                self.state.risk_predictions[nid] = node.collapse_risk
                if node.collapse_risk in risk_counts:
                    risk_counts[node.collapse_risk] += 1
            self.events.metrics.high_risk_zones = risk_counts["HIGH"]
            self.events.metrics.medium_risk_zones = risk_counts["MEDIUM"]
            self.events.metrics.low_risk_zones = risk_counts["LOW"]
            self.events.log(EventType.VULNERABILITY, "Structural vulnerability re-evaluated.")
            # Re-deploy SAR teams based on updated risk
            self.state.sar_deployments = deploy_sar_teams(self.graph)
            self.events.log(EventType.VULNERABILITY, "SAR teams re-deployed to updated vulnerability zones.")

        # Update edge counts in metrics
        self.events.metrics.total_edges_active = self.graph.count_active_edges()
        self.events.metrics.total_edges_blocked = self.graph.count_blocked_edges()

    # ── Aftershock event ────────────────────────────────────

    def _trigger_aftershock(self):
        """Collapse a random active corridor to simulate an aftershock event."""
        active_edges = [(k, e) for k, e in self.graph.edges.items() if not e.blocked]
        if not active_edges:
            return
        key, edge = random.choice(active_edges)
        self.graph.collapse_road(edge.node_a, edge.node_b)
        self.state.aftershock_history.append((self.state.step, edge.node_a, edge.node_b))
        self.events.log(EventType.AFTERSHOCK,
                        f"Aftershock! Corridor between sector {edge.node_a} and {edge.node_b} collapsed!")
        self.events.metrics.total_aftershocks += 1

    # ── Full simulation run ─────────────────────────────────

    def run_full(self) -> SimulationState:
        """
        Run the complete disaster response simulation (initialize + all steps).
        Returns the final SimulationState.
        """
        self.initialize()
        for _ in range(self.state.max_steps):
            self.step_forward()

        self.state.is_complete = True
        self.events.metrics.simulation_end_time = time.time()
        self.events.log(EventType.SIMULATION, "DISASTER RESPONSE SIMULATION COMPLETE.")
        self.events.log(EventType.METRIC, self.events.metrics.summary())

        return self.state
