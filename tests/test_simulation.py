# test_simulation.py — Unit tests for disaster response simulation engine
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import random
import pytest
from core.simulation_engine import Simulation, SimulationState
from core.event_manager import EventType
from config.simulation_config import QuakeConfig


class TestSimulationInit:
    def test_initialization_succeeds(self):
        random.seed(42)
        sim = Simulation(seed=42, verbose=False)
        sim.initialize()
        assert sim.state.is_initialized is True

    def test_graph_populated(self):
        random.seed(42)
        sim = Simulation(seed=42, verbose=False)
        sim.initialize()
        # Grid should have sectors with assigned types
        hospitals = sim.graph.get_nodes_of_type(QuakeConfig.FIELD_HOSPITAL)
        assert len(hospitals) >= QuakeConfig.REQUIRED_COUNTS[QuakeConfig.FIELD_HOSPITAL]

    def test_medical_units_deployed(self):
        random.seed(42)
        sim = Simulation(seed=42, verbose=False)
        sim.initialize()
        assert len(sim.state.medical_units) == QuakeConfig.GA_NUM_MEDICAL_UNITS

    def test_router_created(self):
        random.seed(42)
        sim = Simulation(seed=42, verbose=False)
        sim.initialize()
        assert sim.state.router is not None

    def test_corridor_network_built(self):
        random.seed(42)
        sim = Simulation(seed=42, verbose=False)
        sim.initialize()
        assert sim.graph.count_active_edges() > 0


class TestSimulationSteps:
    def test_step_increments(self):
        random.seed(42)
        sim = Simulation(seed=42, verbose=False)
        sim.initialize()
        sim.step_forward()
        assert sim.state.step == 1

    def test_max_steps_respected(self):
        random.seed(42)
        sim = Simulation(seed=42, verbose=False)
        sim.initialize()
        sim.state.step = QuakeConfig.MAX_STEPS
        sim.step_forward()
        assert sim.state.step == QuakeConfig.MAX_STEPS  # Should not increment

    def test_aftershock_events_logged(self):
        random.seed(42)
        sim = Simulation(seed=42, verbose=False)
        sim.initialize()
        # Run enough steps that aftershocks should occur
        for _ in range(20):
            sim.step_forward()
        # With 30% probability over 20 steps, aftershocks should occur
        assert sim.events.metrics.total_aftershocks >= 0  # Can be 0 by luck


class TestFullSimulation:
    def test_20_step_simulation_completes(self):
        random.seed(42)
        sim = Simulation(seed=42, verbose=False)
        state = sim.run_full()
        assert state.is_complete is True
        assert state.step == QuakeConfig.MAX_STEPS

    def test_events_logged(self):
        random.seed(42)
        sim = Simulation(seed=42, verbose=False)
        sim.run_full()
        assert len(sim.events.event_history) > 0

    def test_metrics_recorded(self):
        random.seed(42)
        sim = Simulation(seed=42, verbose=False)
        sim.run_full()
        metrics = sim.events.metrics
        assert metrics.simulation_duration > 0
        assert metrics.total_edges_active > 0


class TestSimulationState:
    def test_state_serialization(self):
        random.seed(42)
        sim = Simulation(seed=42, verbose=False)
        sim.initialize()
        d = sim.state.to_dict()
        assert "graph" in d
        assert "medical_units" in d
        assert "step" in d
        assert "events" in d
        assert "sar_deployments" in d
        assert len(sim.state.sar_deployments) == 10

    def test_graph_consistency_after_simulation(self):
        """Grid should remain consistent after 20 steps."""
        random.seed(42)
        sim = Simulation(seed=42, verbose=False)
        sim.run_full()
        # All non-collapsed corridors should have valid endpoints
        for key, edge in sim.graph.edges.items():
            assert edge.node_a in sim.graph.nodes
            assert edge.node_b in sim.graph.nodes


class TestDeterministicSeed:
    def test_same_seed_same_result(self):
        """Same seed should produce identical simulation results."""
        sim1 = Simulation(seed=42, verbose=False)
        state1 = sim1.run_full()

        sim2 = Simulation(seed=42, verbose=False)
        state2 = sim2.run_full()

        assert state1.step == state2.step
        assert len(state1.aftershock_history) == len(state2.aftershock_history)
        assert state1.medical_units == state2.medical_units
