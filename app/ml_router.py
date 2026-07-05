from __future__ import annotations

import os
import numpy as np
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import joblib


VALID_MODEL_ROUTES = {
    "fraud_agent",
    "mental_care_agent",
    "daily_care_agent",
}


MODEL_TO_GRAPH_ROUTE = {
    "fraud_agent": "safety_agent",
    "mental_care_agent": "emotional_social_agent",
    "daily_care_agent": "health_daily_care_agent",
}


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "router_best_model.joblib"


class MLRouterError(RuntimeError):
    """Raised when the ML router cannot make a valid prediction."""


def get_model_path() -> Path:
    configured_path = os.getenv("ML_ROUTER_MODEL_PATH", "").strip()
    if not configured_path:
        return DEFAULT_MODEL_PATH

    model_path = Path(configured_path)
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path
    return model_path.resolve()


@lru_cache(maxsize=1)
def load_ml_router():
    """Load the trusted local router model once and cache it."""

    model_path = get_model_path()
    if not model_path.exists():
        raise MLRouterError(f"Router model was not found: {model_path}")

    try:
        model = joblib.load(model_path)
    except Exception as error:
        raise MLRouterError(f"Cannot load router model: {error}") from error

    if not hasattr(model, "predict"):
        raise MLRouterError("Loaded object does not have a predict method.")
    return model


def explain_ml_route(
    text: str,
    top_k: int = 5,
) -> dict[str, Any]:
    """
    Explain one ML router prediction.

    This function is designed for a pipeline containing:
    1. A TF IDF vectorizer named "tfidf"
    2. A linear classifier named "classifier"
    """

    cleaned_text = str(text).strip()

    if not cleaned_text:
        raise MLRouterError(
            "Cannot explain an empty user message."
        )

    model = load_ml_router()

    named_steps = getattr(
        model,
        "named_steps",
        None,
    )

    if named_steps is None:
        raise MLRouterError(
            "The saved model is not a Scikit learn pipeline."
        )

    vectorizer = named_steps.get(
        "tfidf"
    )

    classifier = named_steps.get(
        "classifier"
    )

    if vectorizer is None:
        raise MLRouterError(
            "The pipeline does not contain a tfidf step."
        )

    if classifier is None:
        raise MLRouterError(
            "The pipeline does not contain a classifier step."
        )

    if not hasattr(
        classifier,
        "coef_",
    ):
        raise MLRouterError(
            "The classifier does not expose linear coefficients."
        )

    if not hasattr(
        classifier,
        "decision_function",
    ):
        raise MLRouterError(
            "The classifier does not expose decision scores."
        )

    text_vector = vectorizer.transform(
        [cleaned_text]
    )

    prediction = model.predict(
        [cleaned_text]
    )

    if len(prediction) == 0:
        raise MLRouterError(
            "The model returned no prediction."
        )

    model_label = str(
        prediction[0]
    ).strip()

    if model_label not in VALID_MODEL_ROUTES:
        raise MLRouterError(
            "Model returned an unsupported route: "
            f"{model_label}"
        )

    classes = [
        str(item)
        for item in classifier.classes_
    ]

    if model_label not in classes:
        raise MLRouterError(
            "Predicted label is missing from classifier classes."
        )

    predicted_index = classes.index(
        model_label
    )

    raw_scores = np.asarray(
        classifier.decision_function(
            text_vector
        )
    )

    if raw_scores.ndim == 1:
        if len(classes) == 2:
            positive_score = float(
                raw_scores[0]
            )

            class_scores = np.asarray(
                [
                    -positive_score,
                    positive_score,
                ]
            )
        else:
            class_scores = raw_scores
    else:
        class_scores = raw_scores[0]

    ranked_indices = np.argsort(
        class_scores
    )[::-1]

    alternative_indices = [
        int(index)
        for index in ranked_indices
        if int(index) != predicted_index
    ]

    alternative_index = (
        alternative_indices[0]
        if alternative_indices
        else predicted_index
    )

    alternative_label = classes[
        alternative_index
    ]

    predicted_score = float(
        class_scores[
            predicted_index
        ]
    )

    alternative_score = float(
        class_scores[
            alternative_index
        ]
    )

    decision_margin = (
        predicted_score
        - alternative_score
    )

    coefficient_matrix = np.asarray(
        classifier.coef_
    )

    if (
        len(classes) == 2
        and coefficient_matrix.shape[0] == 1
    ):
        coefficient_matrix = np.vstack(
            [
                -coefficient_matrix[0],
                coefficient_matrix[0],
            ]
        )

    if predicted_index >= coefficient_matrix.shape[0]:
        raise MLRouterError(
            "Cannot match the predicted class "
            "to the classifier coefficients."
        )

    class_coefficients = (
        coefficient_matrix[
            predicted_index
        ]
    )

    feature_names = (
        vectorizer.get_feature_names_out()
    )

    sparse_row = text_vector.tocsr()

    contributions: list[
        dict[str, Any]
    ] = []

    for feature_index, tfidf_value in zip(
        sparse_row.indices,
        sparse_row.data,
    ):
        contribution = float(
            tfidf_value
            * class_coefficients[
                feature_index
            ]
        )

        contributions.append(
            {
                "term": str(
                    feature_names[
                        feature_index
                    ]
                ),
                "tfidf_value": round(
                    float(tfidf_value),
                    4,
                ),
                "contribution": round(
                    contribution,
                    4,
                ),
            }
        )

    supporting_terms = sorted(
        [
            item
            for item in contributions
            if item["contribution"] > 0
        ],
        key=lambda item: item[
            "contribution"
        ],
        reverse=True,
    )[:top_k]

    opposing_terms = sorted(
        [
            item
            for item in contributions
            if item["contribution"] < 0
        ],
        key=lambda item: item[
            "contribution"
        ],
    )[:top_k]

    graph_route = MODEL_TO_GRAPH_ROUTE[
        model_label
    ]

    alternative_graph_route = (
        MODEL_TO_GRAPH_ROUTE.get(
            alternative_label,
            alternative_label,
        )
    )

    return {
        "input_text": cleaned_text,
        "model_label": model_label,
        "graph_route": graph_route,
        "alternative_model_label": (
            alternative_label
        ),
        "alternative_graph_route": (
            alternative_graph_route
        ),
        "predicted_score": round(
            predicted_score,
            4,
        ),
        "alternative_score": round(
            alternative_score,
            4,
        ),
        "decision_margin": round(
            decision_margin,
            4,
        ),
        "margin_is_probability": False,
        "supporting_terms": supporting_terms,
        "opposing_terms": opposing_terms,
    }


def predict_route(text: str) -> str:
    """
    Return only the final graph route.

    Existing code can continue using this function.
    """

    explanation = explain_ml_route(
        text
    )

    return str(
        explanation["graph_route"]
    )


def extract_latest_user_text(state: Mapping[str, Any]) -> str:
    """Find the latest user text from common LangGraph state structures."""

    request = state.get("request")
    request_message = getattr(request, "message", "")
    if isinstance(request_message, str) and request_message.strip():
        return request_message.strip()

    direct_text_keys = (
        "user_message",
        "latest_user_message",
        "message",
        "guest_message",
        "input",
        "text",
        "query",
    )
    for key in direct_text_keys:
        value = state.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    messages = state.get("messages")
    if isinstance(messages, list):
        for message in reversed(messages):
            if isinstance(message, str) and message.strip():
                return message.strip()

            if isinstance(message, dict):
                role = str(message.get("role", message.get("type", ""))).lower()
                content = message.get("content", message.get("text", ""))
                if role in {"user", "human"} and isinstance(content, str) and content.strip():
                    return content.strip()

            role = str(getattr(message, "type", getattr(message, "role", ""))).lower()
            content = getattr(message, "content", None)
            if role in {"user", "human"} and isinstance(content, str) and content.strip():
                return content.strip()

        if messages:
            latest_message = messages[-1]
            if isinstance(latest_message, dict):
                content = latest_message.get("content", latest_message.get("text", ""))
            else:
                content = getattr(latest_message, "content", "")
            if isinstance(content, str) and content.strip():
                return content.strip()

    raise MLRouterError("Cannot find the latest user message in state.")
