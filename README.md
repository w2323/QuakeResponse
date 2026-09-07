# QuakeResponse 🌍🚨

**QuakeResponse** is an advanced, AI-powered earthquake disaster response simulation. It utilizes a suite of robust Artificial Intelligence algorithms to dynamically plan, coordinate, and execute emergency response operations in a city immediately following a massive seismic event.

The project features a purely **Python-based AI engine** (built entirely from scratch using standard libraries) connected to a stunning, interactive **3D React frontend dashboard**. It serves as an exploratory playground for classical and modern AI techniques applied to disaster logistics.

---

## 📖 Table of Contents
1. [Project Overview](#-project-overview)
2. [The Simulation Environment](#-the-simulation-environment)
3. [The 5 Core AI Challenges](#-the-5-core-ai-challenges)
   - [1. Emergency Layout Planning (CSP)](#1-emergency-layout-planning-csp)
   - [2. Emergency Corridors (MST)](#2-emergency-corridors-mst)
   - [3. Vulnerability Assessment (ML)](#3-vulnerability-assessment-ml)
   - [4. Medical Unit Deployment (GA)](#4-medical-unit-deployment-ga)
   - [5. Survivor Rescue Routing (A*)](#5-survivor-rescue-routing-a)
4. [Simulation Event Loop](#-simulation-event-loop)
5. [Technology Stack & Constraints](#-technology-stack--constraints)
6. [Data Structures & Architecture](#-data-structures--architecture)
7. [Setup & Installation Guide](#-setup--installation-guide)
8. [Running the Application](#-running-the-application)
9. [Using the 3D Dashboard](#-using-the-3d-dashboard)
10. [Testing & Validation](#-testing--validation)

---

## 🚁 Project Overview

When a massive earthquake strikes a metropolitan area, emergency services are immediately overwhelmed. Roads collapse, communication networks fail, and critical infrastructure becomes vulnerable. In this chaos, human dispatchers cannot efficiently calculate optimal routes or resource deployments fast enough.

**QuakeResponse** acts as an autonomous "City Brain", stepping in to solve five critical logistical nightmares instantly:

1. Where do we safely place emergency field hospitals and shelters given spatial constraints?
2. Which roads should we clear first to guarantee connectivity across the city?
3. Which buildings are most likely to collapse next due to aftershocks, and where should we deploy Search-and-Rescue (SAR) teams?
4. Where should we park our limited fleet of medical units to minimize response times for civilians?
5. How do rescue teams navigate to trapped survivors when aftershocks are constantly collapsing roads mid-mission?

---

## 🏙️ The Simulation Environment

The city is represented mathematically as a **DisasterGrid**, an undirected weighted graph where:
- **Nodes (Sectors)** represent distinct city blocks (Residential, Industrial, Hospitals, Shelters, etc.).
- **Edges (Corridors)** represent the physical roads connecting these blocks.
- **Edge Weights (Costs)** represent the time/difficulty of traversing a road. These weights are dynamically affected by the structural vulnerability of the adjacent sectors.

The simulation runs over **20 discrete steps** (or turns), simulating the rapid and chaotic early hours of disaster response.

---

## 🧠 The 5 Core AI Challenges

This project implements five major AI algorithms **from scratch**, demonstrating a deep understanding of Constraint Satisfaction, Graph Theory, Machine Learning, Evolutionary Algorithms, and Pathfinding.

### 1. Emergency Layout Planning (CSP)
- **Algorithm**: Constraint Satisfaction Problem (CSP) solver.
- **Techniques Used**: Backtracking Search, Forward Checking (constraint propagation), Minimum Remaining Values (MRV) heuristic, and Minimum-Conflicts fallback.
- **Purpose**: Assign critical roles (`FIELD_HOSPITAL`, `SHELTER`, `GENERATOR_STATION`, `HAZARD_ZONE`) to specific city sectors.
- **Theoretical Execution**: The CSP treats grid coordinates as variables and sector types as domains. It enforces strict spatial rules:
  - Hospitals cannot be placed adjacent to Hazard Zones (industrial chemical plants).
  - Shelters must be adjacent to Generator Stations for power.
  - No two hospitals can be placed next to each other to ensure distributed coverage.
- **Robustness**: If a valid layout cannot be found due to impossible graph constraints, the solver will identify the exact conflict and suggest a minimum-conflict adjustment rather than silently failing.

### 2. Emergency Corridors (MST)
- **Algorithm**: Kruskal’s Minimum Spanning Tree (MST) + Custom Redundancy.
- **Data Structures**: Union-Find (Disjoint Set) with path compression and rank optimization.
- **Purpose**: Synthesize an efficient, unblocked emergency corridor network.
- **Theoretical Execution**: Following an earthquake, clearing all roads is impossible. The MST algorithm guarantees that we can connect every single critical response sector using the absolute minimum total distance of cleared roads. However, a pure tree has no redundancy—if an aftershock destroys one road, the graph splits. Therefore, the algorithm intelligently re-adds a calculated percentage of the shortest unused roads back into the network to create safe, redundant cycles.

### 3. Vulnerability Assessment (ML)
- **Algorithms**: Unsupervised K-Means Clustering + Supervised Decision Tree Classifier.
- **Purpose**: Predict structural collapse risk to deploy SAR teams efficiently.
- **Theoretical Execution**: 
  1. *Feature Engineering*: Sectors are evaluated on their occupancy density (population) and hazard proximity (reciprocal of Dijkstra distance to the nearest hazard zone).
  2. *Clustering*: K-Means (k=3) runs to discover natural vulnerability groupings in the city's topology.
  3. *Synthetic Labels*: Ground-truth labels are generated based on cluster alignment and domain rules.
  4. *Training*: A Decision Tree is trained on the data to classify sectors as `HIGH`, `MEDIUM`, or `LOW` risk. A Decision Tree was specifically chosen over KNN or Neural Networks for its **high interpretability**—it generates clear IF/THEN rules for deployment that humans can audit.
  5. *Deployment*: SAR teams are dispatched to the sectors with the highest priority score (Risk Weight × Population Density).
  6. *Global Impact*: The predicted risk of a sector dynamically increases the cost of navigating through it, directly affecting routing algorithms later.

### 4. Medical Unit Deployment (GA)
- **Algorithm**: Genetic Algorithm (GA).
- **Techniques Used**: Tournament Selection (k=3), Multi-Point Crossover, and Mutation.
- **Purpose**: Optimally place a limited fleet of Medical Units across the city.
- **Theoretical Execution**: 
  - **Chromosomes**: A chromosome is a list of node IDs representing where medical units are parked.
  - **Fitness Function**: Calculates the average Dijkstra distance from all residential sectors to their *nearest* medical unit, heavily weighted by the structural collapse risk of the roads they must travel. (Lower score is better).
  - **Evolution**: The GA spawns an initial population of 50 configurations, ranks them by fitness, breeds the best via crossover, randomly mutates a few locations to prevent local minima, and repeats for 20 generations to find the mathematically optimal deployment strategy.

### 5. Survivor Rescue Routing (A*)
- **Algorithm**: A* (A-Star) Search Algorithm.
- **Heuristic**: Manhattan Distance + Risk Penalties.
- **Purpose**: Plan the safest and fastest route for rescue teams to reach trapped survivors.
- **Theoretical Execution**: The A* algorithm utilizes a Priority Queue to expand nodes that minimize `f(n) = g(n) + h(n)`. Because the simulation includes random **Aftershock** events that permanently collapse roads, the A* router cannot rely on a cached path. At every step, the router checks if its planned route is still viable. If an aftershock has severed a road on the active route, the A* algorithm instantly flushes its cache and dynamically recalculates a new detour mid-mission.

---

## 🔄 Simulation Event Loop

The `simulation_engine.py` orchestrates the flow of time. When you hit **START SIMULATION**, the following occurs in sequence:

**Initialization Phase (Step 0):**
1. CSP runs to distribute hospitals and hazard zones.
2. Kruskal's MST clears the emergency roads.
3. K-Means and the Decision Tree evaluate building vulnerability and update road costs.
4. The GA computes the best parking spots for ambulances based on the new road costs.
5. The A* router plots a course from the main hospital to known trapped survivor locations.

**Execution Phase (Steps 1 to 20):**
1. **Time Advances**: The global step counter increments.
2. **Aftershock Roll**: A random roll (30% probability) determines if an aftershock occurs. If true, a random active road is permanently blocked (`blocked=True`), severing that connection.
3. **Rescue Advancement**: The active Rescue Team moves one node forward along the A* path. If an aftershock blocked their path, they recalculate. If they reach a survivor, they log a rescue.
4. **Dynamic Re-evaluations**:
   - Every 5 steps, the Genetic Algorithm runs again to shift Medical Units to better locations as the road network changes.
   - Every 5 steps, the Decision Tree re-evaluates collapse risk based on the changing environment and repositions SAR teams.

---

## 🛠️ Technology Stack & Constraints

### The "From Scratch" Constraint
A core challenge of this project was the strict prohibition against using external math, machine learning, or graph libraries. 
- **No `scikit-learn`**: The K-Means clustering and Decision Tree classifier are written entirely using raw Python lists and dictionaries. The Decision Tree mathematically calculates Gini Impurity for every possible feature split natively.
- **No `networkx`**: The graph, Kruskal's, Dijkstra's, and A* algorithms are custom-built utilizing standard Python `collections` and `heapq`.

### Software Stack
- **Backend Core**: Python 3.10+
- **Backend API**: FastAPI, Uvicorn
- **Frontend Framework**: React.js, Vite
- **Frontend 3D**: React Three Fiber, Drei (Three.js wrappers)
- **Testing**: Pytest

---

## 🏗️ Data Structures & Architecture

- **`DisasterGrid`**: A custom Graph implementation using Adjacency Lists. Nodes are stored in a dictionary keyed by `node_id`, and edges are stored in a nested dictionary `edges[node_a][node_b] = EdgeObject`.
- **`SimulationState`**: A unified dataclass acting as the Single Source of Truth. It is serialized to JSON on every step via the FastAPI `/state` endpoint, allowing the React frontend to simply act as a dumb renderer of the state.
- **`EventManager`**: An Observer-pattern event bus. Algorithms emit events (e.g., `EventType.AFTERSHOCK`), which are piped to the terminal, logged to a file, and packaged into the JSON state to update the scrolling Activity Log in the frontend UI.

---

## 📂 Project Structure

```text
QuakeResponse/
│
├── core/                        # Core simulation logic and state management
│   ├── city_graph.py            # DisasterGrid definition and Dijkstra implementations
│   ├── event_manager.py         # Centralized logging, metrics, and Event Bus
│   └── simulation_engine.py     # Main engine linking all 5 algorithms together
│
├── challenges/                  # The 5 AI Algorithm Implementations
│   ├── layout_planning.py       # Challenge 1: CSP
│   ├── road_optimization.py     # Challenge 2: Kruskal's MST
│   ├── ambulance_placement.py   # Challenge 3: Genetic Algorithm
│   ├── emergency_routing.py     # Challenge 4: A* Pathfinding
│   └── crime_prediction.py      # Challenge 5: Machine Learning (K-Means & Decision Tree)
│
├── backend/                     # API Bridge
│   └── api/main.py              # FastAPI server exposing the Simulation to the web
│
├── frontend/                    # Interactive 3D React Dashboard
│   ├── src/components/          # React UI Panels and 3D Canvas
│   ├── src/hooks/               # Custom hooks for API polling (useSimulation.js)
│   └── package.json             # Node dependencies
│
└── tests/                       # Comprehensive Pytest test suite (100/100 passing)
```

---

## 🚀 Setup & Installation Guide

To run this project locally, you will need to set up both the Python backend and the Node.js frontend.

### Prerequisites
- **Python 3.10** or higher
- **Node.js 18** or higher
- **npm** (Node Package Manager)

### 1. Backend Setup

Open a terminal in the root directory of the project:

```bash
# 1. Activate the Python virtual environment (Windows)
.\venv\Scripts\activate

# 2. Install required standard dependencies
pip install -r requirements.txt

# 3. Install API server dependencies
pip install fastapi uvicorn pydantic
```

### 2. Frontend Setup

Open a **second terminal** and navigate to the `frontend` folder:

```bash
# 1. Enter the frontend directory
cd frontend

# 2. Install all Node dependencies (React, Three.js, Axios, etc.)
npm install
```

---

## 💻 Running the Application

You must run both the backend API and the frontend dashboard simultaneously in two separate terminals.

### Step 1: Start the Backend API (Terminal 1)
In your first terminal, ensure your virtual environment is active, then run:

```bash
# Start the Uvicorn server on port 8000
uvicorn backend.api.main:app --port 8000
```
*You should see output indicating that `Uvicorn is running on http://0.0.0.0:8000`.*

### Step 2: Start the Frontend Dashboard (Terminal 2)
In your second terminal, inside the `frontend/` directory, run:

```bash
# Start the Vite development server
npm run dev
```
*Vite will output a local URL (e.g., `http://localhost:5173`).*

### Step 3: View the Simulation
Hold `Ctrl` and click the local URL provided by Vite in Terminal 2 to open the QuakeResponse dashboard in your web browser!

---

## 🎮 Using the 3D Dashboard

Once the web application loads:

1. **Initialization**: You will see an empty city grid. 
2. **Start Simulation**: Click the green **"START SIMULATION"** button in the Top Bar. The backend will execute the CSP, MST, ML, and GA algorithms sequentially to build the disaster environment.
3. **Step Forward**: The simulation progresses in turns. Click the **"STEP FORWARD"** button (or the Play icon) to advance time.
4. **Observe Mechanics**:
   - Watch the Rescue Team move along the highlighted path to reach trapped survivors.
   - Look out for red alerts indicating **Aftershocks**. These randomly collapse roads. If a road collapses on the rescue team's path, you will see the A* algorithm instantly recalculate a new route.
   - Every 5 steps, the Vulnerability Assessment (ML) and Medical Unit Deployments (GA) will re-evaluate the city and shift resources dynamically.
5. **Analytics**: The Left and Right panels display real-time metrics, risk counts, GA fitness scores, and system logs generated by the `EventManager`.

---

## 🧪 Testing & Validation

The core AI engine is heavily unit-tested. The test suite guarantees that all algorithms function correctly under extreme edge cases (e.g., disconnected graphs, impossible CSP constraints, empty datasets).

To run the test suite:

```bash
# Ensure you are in the root directory with the venv activated
pytest
```

*All 100 tests should pass flawlessly, validating the integrity of the graph manipulation and machine learning logic.*
