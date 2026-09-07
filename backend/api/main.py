import sys
import os

# Add root directory to python path so we can import core modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from core.simulation_engine import Simulation

app = FastAPI(title="QuakeResponse API", debug=True)

# Allow requests from the Vite frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global simulation instance
sim = Simulation(seed=42, verbose=True)

@app.get("/state")
def get_state():
    """Return the current simulation state."""
    return sim.state.to_dict() if sim.state else {}

@app.get("/graph")
def get_graph():
    """Return the current disaster grid (nodes, edges)."""
    return sim.graph.to_dict() if sim.graph else {}

@app.post("/start")
def start_simulation():
    """Initialize the disaster response simulation."""
    global sim
    sim = Simulation(seed=42, verbose=True)
    sim.initialize()
    return {
        "status": "success",
        "state": sim.state.to_dict(),
        "graph": sim.graph.to_dict()
    }

@app.post("/simulation/step")
def step_simulation():
    """Advance the simulation by one step."""
    if sim.state and not sim.state.is_complete:
        sim.step_forward()
    return {"status": "success", "state": sim.state.to_dict() if sim.state else {}}

@app.post("/simulation/aftershock_node/{node_id}")
def aftershock_node(node_id: int):
    """Manually trigger an aftershock, collapsing corridors near the specified node."""
    if sim.state and not sim.state.is_complete:
        sim._trigger_aftershock(target_node=node_id)
    return {"status": "success", "state": sim.state.to_dict() if sim.state else {}}
