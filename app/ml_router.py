# Explainable ML router that maps user text to graph specialist routes.
from __future__ import annotations

import os
import numpy as np
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import joblib


# Labels emitted by the saved classifier.
VALID_MODEL_ROUTES = {
    "fraud_agent",
    "mental_care_agent",
    "daily_care_agent",
}


# Translate model labels into the route names used by the graph.
MODEL_TO_GRAPH_ROUTE = {
    "fraud_agent": "safety_agent",
    "mental_care_agent": "emotional_social_agent",
    "daily_care_agent": "health_daily_care_agent",
}


# Resolve model paths relative to the project unless an absolute path is given.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "router_best_model.joblib"


# Use a custom error so callers can fall back cleanly.
class MLRouterError(RuntimeError):
    """Raised when the ML router cannot make a valid prediction."""


# Choose the router model path from env or the default models folder.
def get_model_path() -> Path:
    configured_path = os.getenv("ML_ROUTER_MODEL_PATH", "").strip()
    # Empty env var means use the checked-in default path.
    if not configured_path:
        return DEFAULT_MODEL_PATH

    # Relative paths are resolved from the project root for predictable tests.
    model_path = Path(configured_path)
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path
    return model_path.resolve()


# Load and cache the local router model.
@lru_cache(maxsize=1)
def load_ml_router():
    """Load the trusted local router model once and cache it."""

    model_path = get_model_path()
    # Fail clearly when the optional trained model is not present.
    if not model_path.exists():
        raise MLRouterError(f"Router model was not found: {model_path}")

    # joblib load is trusted for the local project model file only.
    try:
        model = joblib.load(model_path)
    except Exception as error:
        raise MLRouterError(f"Cannot load router model: {error}") from error

    # The router must expose predict so downstream code can use it consistently.
    if not hasattr(model, "predict"):
        raise MLRouterError("Loaded object does not have a predict method.")
    return model


# Produce the route plus simple feature-level explanation data.
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

    # Empty text cannot be routed or explained.
    cleaned_text = str(text).strip()

    if not cleaned_text:
        raise MLRouterError(
            "Cannot explain an empty user message."
        )

    # Load the cached sklearn pipeline.
    model = load_ml_router()

    # The explanation path expects a named sklearn pipeline.
    named_steps = getattr(
        model,
        "named_steps",
        None,
    )

    if named_steps is None:
        raise MLRouterError(
        "The saved model is not a Scikit learn pipeline."
        )

    # Pull out the vectorizer and classifier steps used for explanations.
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

    # Linear coefficients are needed to explain which terms helped the route.
    if not hasattr(
        classifier,
        "coef_",
    ):
        raise MLRouterError(
            "The classifier does not expose linear coefficients."
        )

    # Decision scores are needed to calculate the margin from alternatives.
    if not hasattr(
        classifier,
        "decision_function",
    ):
        raise MLRouterError(
            "The classifier does not expose decision scores."
        )

    # Transform text once so prediction and explanation use the same features.
    text_vector = vectorizer.transform(
        [cleaned_text]
    )

    # Ask the model for its route label.
    prediction = model.predict(
        [cleaned_text]
    )

    if len(prediction) == 0:
        raise MLRouterError(
            "The model returned no prediction."
        )

    # Normalize and validate the predicted model label.
    model_label = str(
        prediction[0]
    ).strip()

    if model_label not in VALID_MODEL_ROUTES:
        raise MLRouterError(
            "Model returned an unsupported route: "
            f"{model_label}"
        )

    # Read classifier classes in the same order as decision scores.
    classes = [
        str(item)
        for item in classifier.classes_
    ]

    if model_label not in classes:
        raise MLRouterError(
        "Predicted label is missing from classifier classes."
        )

    # Locate the predicted class so scores and coefficients align.
    predicted_index = classes.index(
        model_label
    )

    # Get raw classifier scores for each class.
    raw_scores = np.asarray(
        classifier.decision_function(
            text_vector
        )
    )

    # Binary sklearn classifiers often return one score, so mirror it to two.
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

    # Rank alternatives by decision score so the margin is explainable.
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

    # Compare the chosen route with the best alternative.
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

    # Convert classifier coefficients into a class-by-feature matrix.
    coefficient_matrix = np.asarray(
        classifier.coef_
    )

    # Binary coefficient matrices need a negative row for the other class.
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

    # Guard against mismatched model internals.
    if predicted_index >= coefficient_matrix.shape[0]:
        raise MLRouterError(
            "Cannot match the predicted class "
            "to the classifier coefficients."
        )

    # Use predicted-class coefficients to score each observed term.
    class_coefficients = (
        coefficient_matrix[
            predicted_index
        ]
    )

    feature_names = (
        vectorizer.get_feature_names_out()
    )

    # Inspect only nonzero terms from this input for a compact explanation.
    sparse_row = text_vector.tocsr()

    contributions: list[
        dict[str, Any]
    ] = []

    # Compute each term's contribution to the predicted class score.
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

    # Positive contributions explain what supported the selected route.
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

    # Negative contributions explain what pushed against the selected route.
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

    # Convert model labels to graph route names for downstream workflow nodes.
    graph_route = MODEL_TO_GRAPH_ROUTE[
        model_label
    ]

    alternative_graph_route = (
        MODEL_TO_GRAPH_ROUTE.get(
            alternative_label,
            alternative_label,
        )
    )

    # Return only serializable fields for API traces and tests.
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


# Predict just the graph route for code that does not need explanations.
def predict_route(text: str) -> str:
    """
    Return only the final graph route.

    Existing code can continue using this function.
    """

    explanation = explain_ml_route(
        text
    )

    # Reuse the explainable path so the route logic stays in one place.
    return str(
        explanation["graph_route"]
    )


# Extract user text from several possible graph state shapes.
def extract_latest_user_text(state: Mapping[str, Any]) -> str:
    """Find the latest user text from common LangGraph state structures."""

    # Prefer the typed request object when the state contains one.
    request = state.get("request")
    request_message = getattr(request, "message", "")
    if isinstance(request_message, str) and request_message.strip():
        return request_message.strip()

    # Then check common direct text keys used by tests and graph variants.
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

    # Finally inspect chat message lists from LangGraph-style state.
    messages = state.get("messages")
    if isinstance(messages, list):
        for message in reversed(messages):
            # Plain strings can appear in simple tests.
            if isinstance(message, str) and message.strip():
                return message.strip()

            # Dictionaries may store role/content or type/text.
            if isinstance(message, dict):
                role = str(message.get("role", message.get("type", ""))).lower()
                content = message.get("content", message.get("text", ""))
                if role in {"user", "human"} and isinstance(content, str) and content.strip():
                    return content.strip()

            # LangChain message objects expose role/type and content attributes.
            role = str(getattr(message, "type", getattr(message, "role", ""))).lower()
            content = getattr(message, "content", None)
            if role in {"user", "human"} and isinstance(content, str) and content.strip():
                return content.strip()

        # If no role is marked, use the latest message content as a fallback.
        if messages:
            latest_message = messages[-1]
            if isinstance(latest_message, dict):
                content = latest_message.get("content", latest_message.get("text", ""))
            else:
                content = getattr(latest_message, "content", "")
            if isinstance(content, str) and content.strip():
                return content.strip()

    raise MLRouterError("Cannot find the latest user message in state.")
