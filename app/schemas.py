# Pydantic models shared by the API, graph, memory, and UI.
from typing import Any, Literal

from pydantic import BaseModel, Field


# Keep risk values constrained so every layer can compare them safely.
RiskLevel = Literal["low", "medium", "high"]


# Validate incoming chat requests before the graph receives them.
class ChatRequest(BaseModel):
    # Vietnamese note: Contract dau vao cua public chat API truoc gateway validation.
    # Require a message while keeping demo defaults for local testing.
    message: str = Field(..., min_length=1)
    user_id: str = Field(default="demo_user", min_length=1)
    conversation_id: str = Field(default="default", min_length=1)
    # Let clients send recent context without forcing server-side memory.
    client_history: list[dict[str, str]] = Field(default_factory=list)


# Capture the router decision before specialist agents run.
class OrchestratorOutput(BaseModel):
    # Each boolean enables one specialist path in the care graph.
    general_reasoning_agent: bool = True
    basic_needs_agent: bool = False
    health_agent: bool = False
    medication_agent: bool = False
    fraud_agent: bool = False
    companion_agent: bool = False
    nutrition_agent: bool = False
    mobility_agent: bool = False
    emergency_agent: bool = False
    technology_agent: bool = False
    caregiver_support_agent: bool = False
    # Intent, priority, and reason explain why those agents were selected.
    intent: str = "general_chat"
    priority: RiskLevel = "low"
    reason: str = ""


# Normalize each specialist result into fields the final response can merge.
class SpecialistResult(BaseModel):
    # Agent identity and risk explain where the advice came from.
    agent: str
    risk: RiskLevel = "low"
    # Signals and actions make the result explainable in the UI.
    signals: list[str] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)
    actions_to_avoid: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    # User-facing response parts are kept separate for safe rendering.
    main_concern: str = ""
    advice: str = ""
    caregiver_alert: str = ""
    next_step: str = ""
    # Reasoning fields preserve what is known and what still needs asking.
    known_facts: list[str] = Field(default_factory=list)
    unknowns: list[str] = Field(default_factory=list)
    follow_up_questions: list[str] = Field(default_factory=list)
    # Keep raw model output available for debugging in controlled traces.
    raw: str = ""


# Return the complete chat result expected by Streamlit and API clients.
class ChatResponse(BaseModel):
    # Vietnamese note: Contract dau ra gom final answer, alert, memory, va technical trace da sanitize.
    # Main answer fields drive the elder-facing response area.
    final_message: str
    active_mode: str = "Mock"
    final_message_source: str = "template"
    # Follow-up fields let the UI explain how much context was reused.
    current_topic: str = ""
    is_follow_up: bool = False
    known_information: list[str] = Field(default_factory=list)
    follow_up_questions: list[str] = Field(default_factory=list)
    specificity_score: float = 0
    missing_information: list[str] = Field(default_factory=list)
    hypotheses: list[dict[str, Any]] = Field(default_factory=list)
    need_more_information: bool = False
    reasoning_summary: str = ""
    xai_simple: str = ""
    caregiver_summary: str = ""
    # Agent and risk fields summarize the safety decision.
    activated_agents: list[str]
    overall_risk: RiskLevel
    main_concern: str
    advice: str
    caregiver_alert: str
    next_step: str
    memory_id: int
    # Confirmation fields block unsafe automatic actions.
    human_confirmation_required: bool = False
    confirmation_reason: str = ""
    # Explainability fields feed the technical trace panels.
    routing_explanation: str = ""
    graph_reasoning: dict[str, Any] = Field(default_factory=dict)
    detected_signals: list[dict[str, str]] = Field(default_factory=list)
    risk_scores: dict[str, int] = Field(default_factory=dict)
    memory_summary: dict[str, Any] = Field(default_factory=dict)
    situation: dict[str, Any] = Field(default_factory=dict)
    care_plan: dict[str, Any] = Field(default_factory=dict)
    developer_state: dict[str, Any] = Field(default_factory=dict)


# Store one persisted chat log row with both summary and raw JSON.
class LogRecord(BaseModel):
    # Database metadata identifies the conversation turn.
    id: int
    timestamp: str
    user_id: str
    message: str
    # Response fields make admin logs readable without opening raw JSON.
    final_message: str = ""
    activated_agents: list[str]
    overall_risk: str
    main_concern: str
    advice: str
    caregiver_alert: str
    next_step: str
    raw_json: dict[str, Any]


# Report optional Kaggle download results to admin callers.
class KaggleDownloadResponse(BaseModel):
    # Separate downloaded, cached, and failed files so the UI can summarize.
    downloaded: list[str]
    cached: list[str]
    errors: list[str]
    csv_files: list[str]
