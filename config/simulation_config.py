# simulation_config.py — Centralized configuration for QuakeResponse
# All tunable parameters in one place. No hardcoded values in modules.


class QuakeConfig:
    """Single source of configuration for the entire QuakeResponse system."""

    # ── Grid ────────────────────────────────────────────────
    GRID_ROWS: int = 15
    GRID_COLS: int = 15

    # ── Road Costs ──────────────────────────────────────────
    COST_STANDARD: float = 1.0
    COST_RESIDENTIAL: float = 0.8
    COST_DIAGONAL: float = 1.4

    # ── Collapse Risk Multipliers (applied to edge costs) ───
    RISK_MULTIPLIER: dict = {
        "HIGH": 2.0,
        "MEDIUM": 1.5,
        "LOW": 1.0,
        "NONE": 1.0,
    }

    # ── Zone Type Constants ─────────────────────────────────
    RESIDENTIAL = "RESIDENTIAL"
    FIELD_HOSPITAL = "FIELD_HOSPITAL"
    SHELTER = "SHELTER"
    HAZARD_ZONE = "HAZARD_ZONE"
    GENERATOR_STATION = "GENERATOR_STATION"
    SUPPLY_DEPOT = "SUPPLY_DEPOT"
    EMPTY = "EMPTY"

    ALL_TYPES = [RESIDENTIAL, FIELD_HOSPITAL, SHELTER, HAZARD_ZONE, GENERATOR_STATION, SUPPLY_DEPOT, EMPTY]

    # ── CSP Resource Zone Requirements (minimum counts) ─────
    REQUIRED_COUNTS: dict = {
        "RESIDENTIAL": 50,
        "FIELD_HOSPITAL": 3,
        "SHELTER": 4,
        "HAZARD_ZONE": 5,
        "GENERATOR_STATION": 2,
        "SUPPLY_DEPOT": 2,
    }

    # ── CSP Constraint Thresholds ───────────────────────────
    HOSPITAL_HOP_LIMIT: int = 3     # C2: residential must be within N blocks of field hospital
    HAZARD_HOP_LIMIT: int = 2       # C3: generator station must be within N blocks of hazard zone

    # ── Emergency Corridor Network (Challenge 2) ────────────
    REDUNDANCY_EDGES: int = 30      # extra corridors added after MST

    # ── Genetic Algorithm (Challenge 3) ─────────────────────
    GA_NUM_MEDICAL_UNITS: int = 3
    GA_POP_SIZE: int = 60
    GA_GENERATIONS: int = 200
    GA_MUTATION_RATE: float = 0.15
    GA_TOURNAMENT_SIZE: int = 5
    GA_ELITISM_COUNT: int = 2

    # ── GA Fast Re-evaluation (mid-simulation) ──────────────
    GA_FAST_POP_SIZE: int = 20
    GA_FAST_GENERATIONS: int = 30

    # ── Survivor Rescue Routing (Challenge 4) ───────────────
    NUM_TRAPPED_SURVIVORS: int = 5  # survivors to rescue per mission

    # ── Structural Vulnerability ML (Challenge 5) ───────────
    KMEANS_K: int = 3
    KMEANS_MAX_ITER: int = 100
    DECISION_TREE_MAX_DEPTH: int = 5
    DECISION_TREE_MIN_SAMPLES: int = 5
    NUM_SAR_TEAMS: int = 10

    # ── Simulation ──────────────────────────────────────────
    MAX_STEPS: int = 20
    AFTERSHOCK_PROBABILITY: float = 0.30
    MEDICAL_REEVAL_INTERVAL: int = 5     # re-evaluate medical units every N steps
    COLLAPSE_REEVAL_INTERVAL: int = 5    # re-evaluate collapse risk every N steps

    # ── Population Values (set after CSP layout) ────────────
    POPULATION_MAP: dict = {
        "RESIDENTIAL": (50, 200),     # random range
        "FIELD_HOSPITAL": 20,
        "SHELTER": 30,
        "HAZARD_ZONE": 10,
        "GENERATOR_STATION": 5,
        "SUPPLY_DEPOT": 5,
        "EMPTY": 0,
    }

    # ── Random Seed (None = non-deterministic) ──────────────
    SEED: int = None
