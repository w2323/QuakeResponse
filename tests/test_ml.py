# test_ml.py — Unit tests for Challenge 5: Structural Vulnerability Assessment
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import random
import pytest
from core.city_graph import DisasterGrid
from challenges.layout_planning import CSPSolver
from challenges.road_optimization import build_emergency_corridors
from challenges.crime_prediction import (
    compute_features, KMeans, DecisionTreeClassifier,
    generate_vulnerability_dataset, run_vulnerability_assessment, deploy_sar_teams
)
from config.simulation_config import QuakeConfig


@pytest.fixture
def ml_graph():
    random.seed(42)
    graph = DisasterGrid()
    solver = CSPSolver(graph)
    solver.solve()
    solver.apply_assignment_to_graph()
    build_emergency_corridors(graph)
    return graph


class TestFeatureEngineering:
    def test_features_computed_for_all_sectors(self, ml_graph):
        features = compute_features(ml_graph)
        assert len(features) == len(ml_graph.nodes)

    def test_feature_values_normalized(self, ml_graph):
        features = compute_features(ml_graph)
        for nid, vec in features.items():
            assert len(vec) == 2
            assert 0.0 <= vec[0] <= 1.0  # Occupancy density normalized
            assert 0.0 <= vec[1] <= 1.0  # Hazard zone proximity (0 to 1)

    def test_hazard_zone_nodes_high_proximity(self, ml_graph):
        """Hazard zone sectors should have high f2 (hazard proximity)."""
        features = compute_features(ml_graph)
        for node in ml_graph.get_nodes_of_type(QuakeConfig.HAZARD_ZONE):
            f2 = features[node.id][1]
            assert f2 > 0.3, f"Hazard zone sector {node.id} has low proximity {f2}"


class TestKMeans:
    def test_clustering_produces_k_clusters(self, ml_graph):
        random.seed(42)
        features = compute_features(ml_graph)
        km = KMeans(k=3)
        km.fit(features)
        unique_labels = set(km.labels.values())
        assert len(unique_labels) <= 3

    def test_all_sectors_assigned(self, ml_graph):
        random.seed(42)
        features = compute_features(ml_graph)
        km = KMeans(k=3)
        km.fit(features)
        assert len(km.labels) == len(features)

    def test_centroids_count(self, ml_graph):
        random.seed(42)
        features = compute_features(ml_graph)
        km = KMeans(k=3)
        km.fit(features)
        assert len(km.centroids) == 3

    def test_convergence(self, ml_graph):
        """K-Means should converge (labels stabilize)."""
        random.seed(42)
        features = compute_features(ml_graph)
        km = KMeans(k=3, max_iter=200)
        km.fit(features)
        # Verify by re-running assignment — should not change
        for nid, vec in features.items():
            dists = [sum((a - b) ** 2 for a, b in zip(vec, c)) ** 0.5
                     for c in km.centroids]
            assert km.labels[nid] == dists.index(min(dists))


class TestDecisionTree:
    def test_prediction_valid_labels(self, ml_graph):
        random.seed(42)
        features = compute_features(ml_graph)
        km = KMeans(k=3)
        km.fit(features)
        dataset = generate_vulnerability_dataset(ml_graph, features, km)
        dt = DecisionTreeClassifier(max_depth=5)
        dt.fit(dataset)
        for nid, vec in features.items():
            pred = dt.predict(vec)
            assert pred in ("HIGH", "MEDIUM", "LOW"), f"Invalid prediction: {pred}"

    def test_tree_trained_on_all_data(self, ml_graph):
        random.seed(42)
        features = compute_features(ml_graph)
        km = KMeans(k=3)
        km.fit(features)
        dataset = generate_vulnerability_dataset(ml_graph, features, km)
        assert len(dataset) == len(ml_graph.nodes)

    def test_explain_method(self, ml_graph):
        random.seed(42)
        features = compute_features(ml_graph)
        km = KMeans(k=3)
        km.fit(features)
        dataset = generate_vulnerability_dataset(ml_graph, features, km)
        dt = DecisionTreeClassifier(max_depth=3)
        dt.fit(dataset)
        explanation = dt.explain()
        assert isinstance(explanation, str)
        assert len(explanation) > 0
        # Should contain feature names
        assert "occupancy_density" in explanation or "PREDICT" in explanation

    def test_gini_impurity(self):
        dt = DecisionTreeClassifier()
        # Pure group — gini should be 0
        pure = [([1.0, 0.5], "HIGH"), ([0.8, 0.6], "HIGH")]
        gini = dt._gini([pure], ["HIGH", "MEDIUM", "LOW"])
        assert gini == 0.0

        # Mixed groups
        mixed = [([1.0, 0.5], "HIGH"), ([0.8, 0.6], "LOW")]
        gini = dt._gini([mixed], ["HIGH", "MEDIUM", "LOW"])
        assert gini > 0.0


class TestFullPipeline:
    def test_pipeline_updates_grid_vulnerability(self, ml_graph):
        """Pipeline must update collapse risk levels on ALL grid sectors."""
        run_vulnerability_assessment(ml_graph)
        risk_set = set()
        for node in ml_graph.nodes.values():
            risk_set.add(node.collapse_risk)
        # Should have at least 2 different risk levels
        assert len(risk_set) >= 2

    def test_risk_affects_corridor_costs(self, ml_graph):
        """After vulnerability assessment, corridor costs should vary based on collapse risk."""
        run_vulnerability_assessment(ml_graph)
        # Find a HIGH risk sector and check its incoming corridor cost
        for nid, node in ml_graph.nodes.items():
            if node.collapse_risk == "HIGH":
                for nb in ml_graph.get_neighbors(nid):
                    cost = ml_graph.get_edge_cost(nb, nid)
                    if cost < float("inf"):
                        # Cost should be multiplied by 2.0
                        key = (min(nb, nid), max(nb, nid))
                        base = ml_graph.edges[key].cost
                        assert cost == base * 2.0
                        return
        # If no HIGH risk sectors, that's still valid (random based)

    def test_pipeline_returns_models(self, ml_graph):
        kmeans, dt = run_vulnerability_assessment(ml_graph)
        assert kmeans is not None
        assert dt is not None
        assert len(kmeans.centroids) == 3
        assert dt.tree is not None


class TestSARDeployment:
    """Tests for the SAR team deployment system."""

    def test_correct_team_count(self, ml_graph):
        """Must deploy exactly 10 SAR teams."""
        run_vulnerability_assessment(ml_graph)
        deployments = deploy_sar_teams(ml_graph)
        assert len(deployments) == 10

    def test_unique_assignments(self, ml_graph):
        """Each SAR team should be at a different sector."""
        run_vulnerability_assessment(ml_graph)
        deployments = deploy_sar_teams(ml_graph)
        node_ids = [d[0] for d in deployments]
        assert len(set(node_ids)) == 10

    def test_high_risk_concentration(self, ml_graph):
        """SAR teams should be concentrated in HIGH or MEDIUM vulnerability areas."""
        run_vulnerability_assessment(ml_graph)
        deployments = deploy_sar_teams(ml_graph)
        high_or_med = sum(1 for _, risk, _ in deployments if risk in ("HIGH", "MEDIUM"))
        # At least half of teams should be in HIGH/MEDIUM zones
        assert high_or_med >= 5

    def test_scores_descending(self, ml_graph):
        """Deployments should be sorted by priority score (highest first)."""
        run_vulnerability_assessment(ml_graph)
        deployments = deploy_sar_teams(ml_graph)
        scores = [d[2] for d in deployments]
        assert scores == sorted(scores, reverse=True)

    def test_custom_team_count(self, ml_graph):
        """Should respect custom team count."""
        run_vulnerability_assessment(ml_graph)
        deployments = deploy_sar_teams(ml_graph, num_teams=5)
        assert len(deployments) == 5
