# ambulance_placement.py — Challenge 3: Medical Unit Deployment via Genetic Algorithm
# Minimax fitness: minimize worst-case survivor-to-medical-unit response time.
#
# CRITICAL FIX: The GA no longer mutates node_type to SUPPLY_DEPOT for medical positions.
# Medical unit positions are stored as a separate list on the SimulationState,
# preserving the CSP-assigned zone types on the grid.
#
# PERFORMANCE FIX: Dijkstra results are cached per node ID. Since the grid
# doesn't change during a GA run, the same node's shortest-path tree is
# reused across all chromosomes. This reduces Dijkstra calls from
# pop_size × generations × num_units (~36,000) to at most 225 unique sectors.
#
# All algorithms implemented from scratch — no external GA frameworks.

import random
from core.city_graph import DisasterGrid
from config.simulation_config import QuakeConfig


class GAMedicalDeployer:
    """
    Genetic Algorithm for optimal medical unit deployment in earthquake zones.

    Objective: Minimize the maximum (worst-case) shortest-path distance
    from any RESIDENTIAL sector to its nearest medical unit.

    Representation:
        Chromosome: list of N node IDs representing medical unit positions
        Fitness: lower is better (minimax distance)

    Operators:
        Selection: Tournament selection (k=5)
        Crossover: Uniform crossover
        Mutation: Random replacement per slot
        Elitism: Top 2 carried forward

    Performance:
        Dijkstra results are cached per node ID. The grid does not change
        during a single GA run, so the same node's shortest-path map is
        reused across all chromosomes that include that node.
    """

    def __init__(self, graph: DisasterGrid,
                 num_units: int = None,
                 pop_size: int = None,
                 generations: int = None,
                 mutation_rate: float = None):
        self.graph = graph
        self.num_units = num_units or QuakeConfig.GA_NUM_MEDICAL_UNITS
        self.pop_size = pop_size or QuakeConfig.GA_POP_SIZE
        self.generations = generations or QuakeConfig.GA_GENERATIONS
        self.mutation_rate = mutation_rate or QuakeConfig.GA_MUTATION_RATE
        self.residential = [n.id for n in graph.get_nodes_of_type(QuakeConfig.RESIDENTIAL)]
        self.all_node_ids = list(graph.nodes.keys())
        self.population: list = []
        self.best_solution: list = None
        self.best_fitness: float = float("inf")

        # Dijkstra cache: node_id -> {dest_id: distance}
        # Since grid state is frozen during GA, we cache all Dijkstra results.
        # At most 225 entries (one per sector), regardless of population or generations.
        self._dijkstra_cache: dict = {}

    # ── Cached Dijkstra ─────────────────────────────────────

    def _cached_dijkstra(self, node_id: int) -> dict:
        """
        Return Dijkstra shortest-path distances from node_id.
        Results are cached because the grid doesn't change during a GA run.
        This is the key performance optimization: instead of computing
        Dijkstra 36,000+ times, we compute it at most 225 times.
        """
        if node_id not in self._dijkstra_cache:
            self._dijkstra_cache[node_id] = self.graph.dijkstra(node_id)
        return self._dijkstra_cache[node_id]

    # ── Population initialization ───────────────────────────

    def initialize_population(self):
        """Create pop_size random chromosomes (each = list of distinct sector IDs)."""
        self.population = []
        for _ in range(self.pop_size):
            chrom = random.sample(self.all_node_ids, self.num_units)
            self.population.append(chrom)

    # ── Fitness evaluation ──────────────────────────────────

    def fitness(self, chromosome: list) -> float:
        """
        Minimax fitness: worst-case response time from any residential sector
        to its nearest medical unit. Lower is better.

        Uses cached Dijkstra which respects collapse risk multipliers and collapsed corridors,
        so vulnerability predictions automatically affect medical unit deployment quality.
        """
        # Get shortest paths from each medical unit position (cached)
        dist_maps = [self._cached_dijkstra(unit_id) for unit_id in chromosome]

        max_dist = 0.0
        for res_id in self.residential:
            # Distance from this residential sector to its nearest medical unit
            min_dist = min(
                dist_maps[i].get(res_id, float("inf"))
                for i in range(self.num_units)
            )
            if min_dist > max_dist:
                max_dist = min_dist
        return max_dist

    # ── Selection ───────────────────────────────────────────

    def tournament_selection(self, fitness_cache: dict,
                             tournament_size: int = None) -> list:
        """Tournament selection: pick best from k random contestants."""
        k = tournament_size or QuakeConfig.GA_TOURNAMENT_SIZE
        contestants = random.sample(self.population, min(k, len(self.population)))
        return min(contestants,
                   key=lambda c: fitness_cache.get(tuple(c), self.fitness(c)))

    # ── Crossover ───────────────────────────────────────────

    def crossover(self, parent1: list, parent2: list) -> tuple:
        """Uniform crossover: for each slot, randomly pick from parent1 or parent2."""
        child1, child2 = [], []
        for i in range(self.num_units):
            if random.random() < 0.5:
                child1.append(parent1[i])
                child2.append(parent2[i])
            else:
                child1.append(parent2[i])
                child2.append(parent1[i])
        child1 = self._fix_duplicates(child1)
        child2 = self._fix_duplicates(child2)
        return child1, child2

    def _fix_duplicates(self, chromosome: list) -> list:
        """Replace duplicate sector IDs with random unique alternatives."""
        seen = set()
        result = []
        for nid in chromosome:
            if nid not in seen:
                seen.add(nid)
                result.append(nid)
            else:
                available = [n for n in self.all_node_ids if n not in seen]
                if available:
                    new_node = random.choice(available)
                    seen.add(new_node)
                    result.append(new_node)
                else:
                    result.append(nid)  # Fallback (won't happen on 225-sector grid)
        return result

    # ── Mutation ────────────────────────────────────────────

    def mutate(self, chromosome: list) -> list:
        """Per-slot mutation: each position has mutation_rate chance of random replacement."""
        result = list(chromosome)
        for i in range(len(result)):
            if random.random() < self.mutation_rate:
                others = [result[j] for j in range(len(result)) if j != i]
                available = [n for n in self.all_node_ids if n not in others]
                if available:
                    result[i] = random.choice(available)
        return result

    # ── Main GA loop ────────────────────────────────────────

    def run(self) -> list:
        """
        Execute the genetic algorithm.
        Returns list of sector IDs for optimal medical unit positions.

        IMPORTANT: Does NOT modify grid sector types.
        Medical unit positions are returned as data, not written to grid.
        """
        self.initialize_population()

        # Evaluate initial population fitness
        fitness_cache = {}
        for c in self.population:
            key = tuple(c)
            if key not in fitness_cache:
                fitness_cache[key] = self.fitness(c)

        for gen in range(self.generations):
            new_population = []

            # Elitism: carry forward top chromosomes
            sorted_pop = sorted(
                self.population,
                key=lambda c: fitness_cache.get(tuple(c), float("inf"))
            )
            for i in range(min(QuakeConfig.GA_ELITISM_COUNT, len(sorted_pop))):
                new_population.append(list(sorted_pop[i]))

            # Track best solution across all generations
            top_fitness = fitness_cache.get(tuple(sorted_pop[0]), float("inf"))
            if top_fitness < self.best_fitness:
                self.best_fitness = top_fitness
                self.best_solution = list(sorted_pop[0])

            # Fill rest of population via selection + crossover + mutation
            while len(new_population) < self.pop_size:
                p1 = self.tournament_selection(fitness_cache)
                p2 = self.tournament_selection(fitness_cache)
                c1, c2 = self.crossover(p1, p2)
                c1 = self.mutate(c1)
                c2 = self.mutate(c2)
                new_population.append(c1)
                if len(new_population) < self.pop_size:
                    new_population.append(c2)

            self.population = new_population

            # Evaluate fitness for new chromosomes
            for c in self.population:
                key = tuple(c)
                if key not in fitness_cache:
                    fitness_cache[key] = self.fitness(c)

        # Final best solution check
        if self.best_solution is None:
            self.best_solution = list(self.population[0])

        return self.best_solution


# ──────────────────────────────────────────────────────────
# Entry points — called by simulation engine
# ──────────────────────────────────────────────────────────
def deploy_medical_units(graph: DisasterGrid) -> list:
    """Full GA run for initial medical unit deployment."""
    deployer = GAMedicalDeployer(graph)
    return deployer.run()


def redeploy_medical_units_fast(graph: DisasterGrid) -> list:
    """Fast GA re-evaluation for mid-simulation repositioning."""
    deployer = GAMedicalDeployer(
        graph,
        pop_size=QuakeConfig.GA_FAST_POP_SIZE,
        generations=QuakeConfig.GA_FAST_GENERATIONS,
    )
    return deployer.run()
