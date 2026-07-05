# Unit tests for the explainable ML router and graph integration.
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import joblib
import numpy as np

from app.config import get_settings
from app.graph import run_elderguard_workflow
from app.ml_router import explain_ml_route, load_ml_router, predict_route
from app.schemas import ChatRequest


# Fake classifier labels mirror the saved router labels.
CLASSES = [
    "fraud_agent",
    "mental_care_agent",
    "daily_care_agent",
]


# Minimal sparse vector that behaves like the vectorizer output used by sklearn.
class FakeSparseVector:
    def __init__(self, indices: list[int], data: list[float]):
        self.indices = np.asarray(indices, dtype=int)
        self.data = np.asarray(data, dtype=float)

    # The router calls tocsr, so return self for the fake sparse row.
    def tocsr(self):
        return self


# Fake TF-IDF vectorizer that marks known words with value 1.0.
class FakeVectorizer:
    def __init__(self):
        self.feature_names = np.asarray(
            [
                "bank",
                "password",
                "hopeless",
                "anxious",
                "dizzy",
                "standing",
            ]
        )
        self.vocabulary = {
            term: index
            for index, term in enumerate(self.feature_names)
        }

    # Transform one text input into a fake sparse vector.
    def transform(self, values):
        text = str(values[0]).lower()
        indices: list[int] = []
        data: list[float] = []

        for term, index in self.vocabulary.items():
            if term in text:
                indices.append(index)
                data.append(1.0)

        return FakeSparseVector(indices, data)

    # Return feature names so explanations can name supporting terms.
    def get_feature_names_out(self):
        return self.feature_names


# Fake linear classifier that exposes coefficients and decision scores.
class FakeClassifier:
    def __init__(self, forced_route: str | None = None):
        self.classes_ = np.asarray(CLASSES)
        self.forced_route = forced_route
        self.coef_ = np.asarray(
            [
                [1.0, 1.0, -0.2, -0.2, -0.1, -0.1],
                [-0.2, -0.2, 1.0, 1.0, -0.1, -0.1],
                [-0.1, -0.1, -0.2, -0.2, 1.0, 1.0],
            ]
        )

    # Score each class from the fake vector and optionally force one route.
    def decision_function(self, text_vector):
        scores = self.coef_[:, text_vector.indices] @ text_vector.data

        if self.forced_route:
            forced_index = CLASSES.index(self.forced_route)
            scores = np.asarray(scores, dtype=float)
            scores[forced_index] = max(scores.max(initial=0), 0) + 2.0

        return np.asarray([scores])


# Fake sklearn pipeline with named steps used by explain_ml_route.
class FakeRouterPipeline:
    def __init__(self, forced_route: str | None = None):
        self.named_steps = {
            "tfidf": FakeVectorizer(),
            "classifier": FakeClassifier(forced_route),
        }

    # Pick the class with the highest fake decision score.
    def predict(self, values):
        vector = self.named_steps["tfidf"].transform(values)
        scores = self.named_steps["classifier"].decision_function(vector)[0]
        return [CLASSES[int(np.argmax(scores))]]


# Small assertion helper keeps standalone output readable.
def assert_equal(actual, expected, label: str) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


# Temporarily write a fake model and point the router at it.
def with_fake_model(model, callback):
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "router_best_model.joblib"
        joblib.dump(model, model_path)
        os.environ["ML_ROUTER_MODEL_PATH"] = str(model_path)
        load_ml_router.cache_clear()
        try:
            return callback()
        finally:
            # Clear caches and env so tests do not leak into each other.
            load_ml_router.cache_clear()
            os.environ.pop("ML_ROUTER_MODEL_PATH", None)


# Predict through a fake forced model route.
def predict_with_fake_model(model_route: str, text: str) -> str:
    return with_fake_model(
        FakeRouterPipeline(model_route),
        lambda: predict_route(text),
    )


# Explain a text through the fake model.
def explain_with_fake_model(text: str) -> dict:
    return with_fake_model(
        FakeRouterPipeline(),
        lambda: explain_ml_route(text),
    )


# Verify saved model labels map to graph broad-agent routes.
def test_predict_route_maps_model_labels_to_graph_routes():
    cases = [
        ("fraud_agent", "Someone asked me for my bank password.", "safety_agent"),
        ("mental_care_agent", "I feel hopeless and anxious every day.", "emotional_social_agent"),
        ("daily_care_agent", "I feel dizzy when I stand up.", "health_daily_care_agent"),
    ]

    for model_route, text, expected_graph_route in cases:
        predicted = predict_with_fake_model(model_route, text)
        assert_equal(predicted, expected_graph_route, text)


# Verify the graph can use an ML-selected route when rules are weak.
def test_langgraph_uses_ml_selected_broad_agent():
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "router_best_model.joblib"
        joblib.dump(FakeRouterPipeline("fraud_agent"), model_path)
        os.environ["LLM_MODE"] = "mock"
        os.environ["USE_ML_ROUTER"] = "true"
        os.environ["ML_ROUTER_MODEL_PATH"] = str(model_path)
        get_settings.cache_clear()
        load_ml_router.cache_clear()
        try:
            # A neutral message lets the ML route take priority.
            response = run_elderguard_workflow(
                ChatRequest(
                    message="I need help with a normal daily care question.",
                    user_id="ml_router_test",
                    conversation_id="fake_model_graph",
                )
            )
            assert_equal(response.developer_state["router_decision"]["router_source"], "ml", "router source")
            assert_equal(response.developer_state["router_decision"]["selected_agent"], "safety_agent", "selected agent")
            assert_equal(response.activated_agents, ["safety_agent"], "activated agents")
        finally:
            # Restore router settings after the integration test.
            load_ml_router.cache_clear()
            get_settings.cache_clear()
            os.environ.pop("ML_ROUTER_MODEL_PATH", None)
            os.environ["USE_ML_ROUTER"] = "false"


# Verify explanation output includes model label, route, margin, and terms.
def test_explain_ml_route_returns_xai_fields():
    result = explain_with_fake_model(
        "I feel dizzy after standing up"
    )

    assert result["model_label"] == "daily_care_agent"
    assert result["graph_route"] == "health_daily_care_agent"
    assert result["margin_is_probability"] is False
    assert isinstance(result["supporting_terms"], list)
    assert "decision_margin" in result


# Verify fraud-like text routes to the safety broad agent.
def test_fraud_explanation_route():
    result = explain_with_fake_model(
        "Someone asked for my bank password"
    )

    assert result["model_label"] == "fraud_agent"
    assert result["graph_route"] == "safety_agent"


# Verify mental-care text routes to emotional/social support.
def test_mental_care_explanation_route():
    result = explain_with_fake_model(
        "I feel hopeless and anxious"
    )

    assert result["model_label"] == "mental_care_agent"
    assert result["graph_route"] == "emotional_social_agent"


# Run the router tests as a standalone script.
def main() -> None:
    test_predict_route_maps_model_labels_to_graph_routes()
    print("PASS: model labels map to graph routes")
    test_langgraph_uses_ml_selected_broad_agent()
    print("PASS: LangGraph used ML-selected broad agent")
    test_explain_ml_route_returns_xai_fields()
    print("PASS: explanation includes XAI fields")
    test_fraud_explanation_route()
    print("PASS: fraud explanation route")
    test_mental_care_explanation_route()
    print("PASS: mental care explanation route")


# Allow manual execution without pytest.
if __name__ == "__main__":
    main()
