# QuakeResponse 🌍🚨

**QuakeResponse** is an AI-powered earthquake disaster response simulation. It uses a suite of Artificial Intelligence algorithms to dynamically plan, coordinate, and execute emergency response operations in a city immediately following a massive seismic event. 

The project features a purely Python-based AI engine connected to a beautiful interactive 3D React frontend.

---

## 🧠 Core AI Challenges

This project implements five major AI challenges from scratch (without external ML libraries) to form a complete response pipeline:

1. **Emergency Layout Planning (CSP)**
   - *Algorithm*: Constraint Satisfaction Problem (CSP) solver with Backtracking, Forward Checking, and the Minimum Remaining Values (MRV) heuristic.
   - *Goal*: Assign critical roles (Field Hospitals, Shelters, Generator Stations, Hazard Zones) to city sectors while strictly respecting adjacency constraints (e.g., hospitals cannot be placed next to hazard zones).

2. **Emergency Corridors (Graph Algorithms)**
   - *Algorithm*: Kruskal’s Minimum Spanning Tree (MST) + Custom Redundancy.
   - *Goal*: Synthesize an efficient, unblocked emergency corridor network connecting all critical response sectors to guarantee access for rescue units.

3. **Vulnerability Assessment (Machine Learning)**
   - *Algorithm*: Unsupervised K-Means Clustering + Supervised Decision Tree Classifier.
   - *Goal*: Group sectors based on occupancy density and hazard proximity, generate synthetic ground truth, and train an explainable Decision Tree to predict structural collapse risk (`HIGH`, `MEDIUM`, `LOW`). This directly dictates the deployment of Search-and-Rescue (SAR) teams.

4. **Medical Unit Deployment (Genetic Algorithm)**
   - *Algorithm*: Genetic Algorithm (GA) with Tournament Selection, Multi-Point Crossover, and Mutation.
   - *Goal*: Optimally place limited medical units across the city, minimizing the risk-weighted distance between civilians and their closest medical facility.

5. **Survivor Rescue Routing (Pathfinding)**
   - *Algorithm*: A* Search Algorithm.
   - *Goal*: Plan the safest and fastest route for rescue teams to reach trapped survivors. The router dynamically recalculates paths if random **Aftershocks** collapse emergency corridors mid-mission.

---

## 🛠️ Technology Stack

- **Backend**: Python 3, FastAPI, Uvicorn (No external ML libraries allowed; strictly built from standard libraries).
- **Frontend**: React, Vite, Axios, 3D Canvas.
- **Testing**: Pytest (100% core logic coverage).

---

## 🚀 Setup & Installation

You will need **two separate terminal windows** to run the backend and the frontend.

### 1. Backend Setup (Python)

Open a terminal in the project root directory:

```bash
# Activate the virtual environment
.\venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
pip install fastapi uvicorn
```

### 2. Frontend Setup (Node.js)

Open a second terminal and navigate to the frontend directory:

```bash
cd frontend

# Install Node modules
npm install
```

---

## 💻 Running the Simulation

Once dependencies are installed, you can launch the application:

### Start the Backend API (Terminal 1)
```bash
# From the root directory (with venv activated)
uvicorn backend.api.main:app --port 8000
```
*The backend will now be serving data at `http://localhost:8000`.*

### Start the React Frontend (Terminal 2)
```bash
# From the frontend directory
npm run dev
```
*Vite will provide a local URL (usually `http://localhost:5173` or `http://localhost:5174`). Click it to open the 3D dashboard in your browser!*

---

## 🧪 Running Tests

The AI core features a comprehensive testing suite simulating extreme edge cases.

```bash
# Run all tests using pytest
pytest
```

---

## 🎮 How to use the Dashboard

1. Wait for the initial simulation data to load into the dashboard.
2. Click **START SIMULATION** to run the 5-step AI initialization pipeline (CSP → MST → ML → GA → A*).
3. Use the **STEP FORWARD** button or hit **PLAY** to watch the disaster scenario unfold over 20 turns.
4. Watch out for random **Aftershocks** collapsing roads and forcing your routing algorithms to recalculate on the fly!
