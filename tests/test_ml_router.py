from __future__ import annotations

import os
import tempfile
from pathlib import Path

import joblib

from app.config import get_settings
from app.graph import run_elderguard_workflow
from app.ml_router import load_ml_router, predict_route
from app.schemas import ChatRequest


class FakeRouterModel:
    def __init__(self, route: str):
        self.route = route

    def predict(self, values):
        return [self.route for _ in values]


def assert_equal(actual, expected, label: str) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def predict_with_fake_model(model_route: str, text: str) -> str:
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "router_best_model.joblib"
        joblib.dump(FakeRouterModel(model_route), model_path)
        os.environ["ML_ROUTER_MODEL_PATH"] = str(model_path)
        load_ml_router.cache_clear()
        try:
            return predict_route(text)
        finally:
            load_ml_router.cache_clear()
            os.environ.pop("ML_ROUTER_MODEL_PATH", None)


def main() -> None:
    cases = [
        ("fraud_agent", "Someone asked me for my bank password.", "safety_agent"),
        ("mental_care_agent", "I feel hopeless and anxious every day.", "emotional_social_agent"),
        ("daily_care_agent", "I feel dizzy when I stand up.", "health_daily_care_agent"),
    ]
    for model_route, text, expected_graph_route in cases:
        predicted = predict_with_fake_model(model_route, text)
        assert_equal(predicted, expected_graph_route, text)
        print(f"PASS: {model_route} -> {predicted}")

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "router_best_model.joblib"
        joblib.dump(FakeRouterModel("fraud_agent"), model_path)
        os.environ["LLM_MODE"] = "mock"
        os.environ["USE_ML_ROUTER"] = "true"
        os.environ["ML_ROUTER_MODEL_PATH"] = str(model_path)
        get_settings.cache_clear()
        load_ml_router.cache_clear()
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
        print("PASS: LangGraph used ML-selected broad agent")
        load_ml_router.cache_clear()
        get_settings.cache_clear()
        os.environ.pop("ML_ROUTER_MODEL_PATH", None)
        os.environ["USE_ML_ROUTER"] = "false"


if __name__ == "__main__":
    main()
