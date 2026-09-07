// ═══════════════════════════════════════════════════════════
// CITYMIND — useSimulation Hook
// Manages backend API connection and simulation state.
// React state updates ONLY on meaningful events (not in rAF).
// ═══════════════════════════════════════════════════════════

import { useState, useEffect, useCallback, useMemo } from 'react';
import { getState, getGraph, startSimulation, stepSimulation, aftershockNode } from '../api/quakeresponse';

export function useSimulation() {
  const [state, setState] = useState(null);
  const [graph, setGraph] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [connected, setConnected] = useState(false);
  const [stepping, setStepping] = useState(false);
  const [starting, setStarting] = useState(false);

  const loadData = useCallback(async () => {
    try {
      setError(null);
      const [s, g] = await Promise.all([getState(), getGraph()]);
      setState(s);
      setGraph(g);
      setConnected(true);
      setLoading(false);
    } catch (err) {
      setError('Cannot connect to QuakeResponse backend. Ensure uvicorn is running on port 8000.');
      setConnected(false);
      setLoading(false);
    }
  }, []);

  const handleStart = useCallback(async () => {
    try {
      setStarting(true);
      const result = await startSimulation();
      if (result.state) setState(result.state);
      const g = await getGraph();
      setGraph(g);
      setConnected(true);
    } catch (err) {
      setError('Failed to start simulation');
    } finally {
      setStarting(false);
    }
  }, []);

  const handleStep = useCallback(async () => {
    try {
      setStepping(true);
      const result = await stepSimulation();
      if (result.state) setState(result.state);
      else { const s = await getState(); setState(s); }
    } catch (err) {
      setError('Failed to step simulation');
    } finally {
      setStepping(false);
    }
  }, []);

  const handleAftershockNode = useCallback(async (nodeId) => {
    try {
      setStepping(true); // use stepping state to show loading
      const result = await aftershockNode(nodeId);
      if (result.state) setState(result.state);
      else { const s = await getState(); setState(s); }
    } catch (err) {
      setError('Failed to trigger aftershock');
    } finally {
      setStepping(false);
    }
  }, []);

  useEffect(() => { loadData(); }, [loadData]);

  const stats = useMemo(() => {
    if (!state || !graph) return null;
    const nodes = graph.nodes || {};
    const nodeList = Object.values(nodes);
    const typeCounts = {};
    nodeList.forEach(n => {
      const t = n.node_type || 'EMPTY';
      typeCounts[t] = (typeCounts[t] || 0) + 1;
    });

    const riskPredictions = state.risk_predictions || {};
    const riskCounts = { HIGH: 0, MEDIUM: 0, LOW: 0 };
    Object.values(riskPredictions).forEach(r => { if (r in riskCounts) riskCounts[r]++; });

    const edgeList = [];
    const edgesObj = graph.edges || {};
    Object.values(edgesObj).forEach(e => {
      edgeList.push({ nodeA: e.node_a, nodeB: e.node_b, cost: e.cost, blocked: e.blocked });
    });

    return {
      totalNodes: nodeList.length,
      typeCounts,
      riskCounts,
      residential: typeCounts['RESIDENTIAL'] || 0,
      hospitals: typeCounts['FIELD_HOSPITAL'] || 0,
      schools: typeCounts['SHELTER'] || 0,
      industrial: typeCounts['HAZARD_ZONE'] || 0,
      powerPlants: typeCounts['GENERATOR_STATION'] || 0,
      depots: typeCounts['SUPPLY_DEPOT'] || 0,
      sar: state.sar_deployments || [],
      medical_units: state.medical_units || [],
      aftershocks: state.aftershock_history || [],
      step: state.step || 0,
      maxSteps: state.max_steps || 20,
      isComplete: state.is_complete || false,
      events: state.events?.events || [],
      metrics: state.events?.metrics || {},
      router: state.router || {},
      activeRoutes: state.active_routes || [],
      civiliansRemaining: state.router?.survivors_remaining || [],
      edgeList,
    };
  }, [state, graph]);

  return { state, graph, stats, loading, error, connected, stepping, starting, handleStart, handleStep, handleAftershockNode, loadData };
}
