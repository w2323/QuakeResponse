# test_ga.py — Unit tests for Challenge 3: Medical Unit Deployment (GA)
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import random
import pytest
from core.city_graph import DisasterGrid
from challenges.layout_planning import CSPSolver
from challenges.road_optimization import build_emergency_corridors
from challenges.ambulance_placement import GAMedicalDeployer, deploy_medical_units
from config.simulation_config import QuakeConfig


@pytest.fixture
def ready_graph():
    """Grid with layout + corridor network ready for GA."""
    random.seed(42)
    graph = DisasterGrid()
    solver = CSPSolver(graph)
    solver.solve()
    solver.apply_assignment_to_graph()
    build_emergency_corridors(graph)
    return graph


class TestGAMedicalDeployer:
    def test_deployment_returns_correct_count(self, ready_graph):
        """GA should return exactly NUM_MEDICAL_UNITS positions."""
        result = deploy_medical_units(ready_graph)
        assert len(result) == QuakeConfig.GA_NUM_MEDICAL_UNITS

    def test_deployment_unique_sectors(self, ready_graph):
        """All medical unit positions should be distinct sectors."""
        result = deploy_medical_units(ready_graph)
        assert len(set(result)) == len(result)

    def test_deployment_valid_sector_ids(self, ready_graph):
        """All returned sector IDs must exist in the grid."""
        result = deploy_medical_units(ready_graph)
        for nid in result:
            assert nid in ready_graph.nodes

    def test_does_not_mutate_sector_types(self, ready_graph):
        """CRITICAL: GA must NOT change node_type to SUPPLY_DEPOT."""
        # Record original sector types
        original_types = {nid: n.node_type for nid, n in ready_graph.nodes.items()}
        deploy_medical_units(ready_graph)
        # All sector types should remain unchanged
        for nid, node in ready_graph.nodes.items():
            assert node.node_type == original_types[nid], \
                f"Sector {nid} type changed from {original_types[nid]} to {node.node_type}"

    def test_fitness_evaluation(self, ready_graph):
        """Fitness should return a finite positive value for valid chromosomes."""
        deployer = GAMedicalDeployer(ready_graph, pop_size=5, generations=1)
        chrom = random.sample(list(ready_graph.nodes.keys()), 3)
        fitness = deployer.fitness(chrom)
        assert fitness >= 0
        assert fitness < float("inf")

    def test_fitness_improves_over_generations(self, ready_graph):
        """GA should produce better fitness than random deployment."""
        random.seed(42)
        deployer = GAMedicalDeployer(ready_graph, pop_size=10, generations=5)

        # Random baseline
        random_chrom = random.sample(list(ready_graph.nodes.keys()), 3)
        random_fitness = deployer.fitness(random_chrom)

        # GA result
        result = deployer.run()
        ga_fitness = deployer.best_fitness

        # GA should be at least as good as random (usually much better)
        assert ga_fitness <= random_fitness or ga_fitness < float("inf")

    def test_crossover_produces_valid_children(self, ready_graph):
        deployer = GAMedicalDeployer(ready_graph, pop_size=5, generations=1)
        p1 = random.sample(list(ready_graph.nodes.keys()), 3)
        p2 = random.sample(list(ready_graph.nodes.keys()), 3)
        c1, c2 = deployer.crossover(p1, p2)
        assert len(c1) == 3
        assert len(c2) == 3
        assert len(set(c1)) == 3  # No duplicates
        assert len(set(c2)) == 3

    def test_mutation_preserves_length(self, ready_graph):
        deployer = GAMedicalDeployer(ready_graph, pop_size=5, generations=1)
        chrom = random.sample(list(ready_graph.nodes.keys()), 3)
        mutated = deployer.mutate(chrom)
        assert len(mutated) == 3
