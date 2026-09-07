# conftest.py — Shared test fixtures and configuration
# Overrides GA parameters for faster test execution.
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.simulation_config import QuakeConfig

# Override GA defaults for test speed
# Production: 60 pop, 200 gen → Test: 10 pop, 5 gen
QuakeConfig.GA_POP_SIZE = 10
QuakeConfig.GA_GENERATIONS = 5
QuakeConfig.GA_FAST_POP_SIZE = 6
QuakeConfig.GA_FAST_GENERATIONS = 3
