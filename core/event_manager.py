# event_manager.py — Centralized logging, event bus, and metrics for QuakeResponse
# Provides structured logging with EventType enum, simulation metrics tracking,
# and an observer-pattern event bus for decoupled module communication.

import logging
import os
import time
from enum import Enum, auto
from datetime import datetime


# ──────────────────────────────────────────────────────────
# Event Types — typed enum instead of raw strings
# ──────────────────────────────────────────────────────────
class EventType(Enum):
    LAYOUT = auto()        # CSP resource zone planning events
    ROADS = auto()         # Emergency corridor construction
    AFTERSHOCK = auto()    # Aftershock road collapse events
    VULNERABILITY = auto() # Structural vulnerability assessment
    MEDICAL = auto()       # Medical unit deployment / repositioning
    ROUTING = auto()       # A* rescue route planning / replanning
    SIMULATION = auto()    # General simulation lifecycle
    GRAPH = auto()         # Grid structural changes
    METRIC = auto()        # Performance / metrics events


# ──────────────────────────────────────────────────────────
# Simulation Metrics — tracks key performance indicators
# ──────────────────────────────────────────────────────────
class SimulationMetrics:
    """Collects and stores disaster response KPIs for analytics and presentation."""

    def __init__(self):
        self.total_aftershocks: int = 0
        self.total_reroutes: int = 0
        self.survivors_rescued: int = 0
        self.survivors_unreachable: int = 0
        self.high_risk_zones: int = 0
        self.medium_risk_zones: int = 0
        self.low_risk_zones: int = 0
        self.medical_repositions: int = 0
        self.avg_response_time: float = 0.0
        self.worst_response_time: float = 0.0
        self.disconnected_sectors: int = 0
        self.total_edges_active: int = 0
        self.total_edges_blocked: int = 0
        self.csp_constraints_checked: int = 0
        self.ga_best_fitness: float = float("inf")
        self.simulation_start_time: float = 0.0
        self.simulation_end_time: float = 0.0
        self._response_times: list = []

    def record_response_time(self, time_val: float):
        """Record a single response time measurement."""
        self._response_times.append(time_val)
        self.avg_response_time = sum(self._response_times) / len(self._response_times)
        self.worst_response_time = max(self._response_times)

    @property
    def simulation_duration(self) -> float:
        if self.simulation_end_time > 0:
            return self.simulation_end_time - self.simulation_start_time
        return time.time() - self.simulation_start_time

    def to_dict(self) -> dict:
        """Serialize metrics for API / frontend consumption."""
        return {
            "total_aftershocks": self.total_aftershocks,
            "total_reroutes": self.total_reroutes,
            "survivors_rescued": self.survivors_rescued,
            "survivors_unreachable": self.survivors_unreachable,
            "high_risk_zones": self.high_risk_zones,
            "medium_risk_zones": self.medium_risk_zones,
            "low_risk_zones": self.low_risk_zones,
            "medical_repositions": self.medical_repositions,
            "avg_response_time": round(self.avg_response_time, 3),
            "worst_response_time": round(self.worst_response_time, 3),
            "disconnected_sectors": self.disconnected_sectors,
            "total_edges_active": self.total_edges_active,
            "total_edges_blocked": self.total_edges_blocked,
            "ga_best_fitness": round(self.ga_best_fitness, 3) if self.ga_best_fitness != float('inf') else None,
            "simulation_duration_sec": round(self.simulation_duration, 2),
        }

    def summary(self) -> str:
        """Human-readable disaster response metrics summary."""
        lines = [
            "=" * 55,
            "  DISASTER RESPONSE METRICS SUMMARY",
            "=" * 55,
            f"  Aftershocks triggered:  {self.total_aftershocks}",
            f"  Reroutes triggered:     {self.total_reroutes}",
            f"  Survivors rescued:      {self.survivors_rescued}",
            f"  Survivors unreachable:  {self.survivors_unreachable}",
            f"  Medical repositions:    {self.medical_repositions}",
            f"  Avg response time:      {self.avg_response_time:.3f}",
            f"  Worst response time:    {self.worst_response_time:.3f}",
            f"  GA best fitness:        {self.ga_best_fitness:.3f}",
            f"  Risk zones (H/M/L):     {self.high_risk_zones}/{self.medium_risk_zones}/{self.low_risk_zones}",
            f"  Corridors active/blocked: {self.total_edges_active}/{self.total_edges_blocked}",
            f"  Disconnected sectors:   {self.disconnected_sectors}",
            f"  Duration:               {self.simulation_duration:.2f}s",
            "=" * 55,
        ]
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────
# Event Manager — centralized logging and event distribution
# ──────────────────────────────────────────────────────────
class EventManager:
    """
    Centralized event logging and distribution system.

    - Logs to both console and file
    - Stores structured event history
    - Provides observer pattern for event subscribers
    - Tracks disaster response metrics
    """

    def __init__(self, log_dir: str = "logs", verbose: bool = True):
        self.metrics = SimulationMetrics()
        self.event_history: list = []  # List of (step, EventType, message) tuples
        self._observers: dict = {}     # EventType -> list of callbacks
        self._current_step: int = 0
        self._verbose = verbose

        # Set up Python logging
        self._logger = logging.getLogger("QuakeResponse")
        self._logger.setLevel(logging.DEBUG)
        self._logger.handlers.clear()

        # Console handler
        console = logging.StreamHandler()
        console.setLevel(logging.INFO if not verbose else logging.DEBUG)
        console_fmt = logging.Formatter("%(message)s")
        console.setFormatter(console_fmt)
        self._logger.addHandler(console)

        # File handler
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"quakeresponse_{timestamp}.log")
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")
        file_handler.setFormatter(file_fmt)
        self._logger.addHandler(file_handler)

        self.log_file_path = log_file

    def set_step(self, step: int):
        """Update the current simulation step for log context."""
        self._current_step = step

    def log(self, event_type: EventType, message: str, level: str = "info"):
        """
        Log a structured event.

        Args:
            event_type: Category of the event
            message: Human-readable description
            level: Log level ('debug', 'info', 'warning', 'error')
        """
        step_str = f"STEP {self._current_step:02d}" if self._current_step > 0 else "INIT"
        formatted = f"[{step_str}] [{event_type.name:14s}] {message}"

        # Store in history
        self.event_history.append((self._current_step, event_type, message))

        # Log via Python logging
        log_fn = getattr(self._logger, level, self._logger.info)
        log_fn(formatted)

        # Notify observers
        self._notify(event_type, message)

    def subscribe(self, event_type: EventType, callback):
        """Register a callback for a specific event type."""
        if event_type not in self._observers:
            self._observers[event_type] = []
        self._observers[event_type].append(callback)

    def _notify(self, event_type: EventType, message: str):
        """Notify all subscribers of an event."""
        for cb in self._observers.get(event_type, []):
            cb(self._current_step, event_type, message)

    def get_events(self, event_type: EventType = None, step: int = None) -> list:
        """Query event history with optional filters."""
        results = self.event_history
        if event_type is not None:
            results = [(s, t, m) for s, t, m in results if t == event_type]
        if step is not None:
            results = [(s, t, m) for s, t, m in results if s == step]
        return results

    def to_dict(self) -> dict:
        """Serialize the full event log for API consumption."""
        return {
            "events": [
                {"step": s, "type": t.name, "message": m}
                for s, t, m in self.event_history
            ],
            "metrics": self.metrics.to_dict(),
        }
