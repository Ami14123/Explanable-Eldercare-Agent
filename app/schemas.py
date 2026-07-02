from typing import Any, Literal

from pydantic import BaseModel, Field


RiskLevel = Literal["low", "medium", "high"]


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    user_id: str = Field(default="demo_user", min_length=1)
    conversation_id: str = Field(default="default", min_length=1)
    client_history: list[dict[str, str]] = Field(default_factory=list)


class OrchestratorOutput(BaseModel):
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
    intent: str = "general_chat"
    priority: RiskLevel = "low"
    reason: str = ""


class SpecialistResult(BaseModel):
    agent: str
    risk: RiskLevel = "low"
    signals: list[str] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)
    actions_to_avoid: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    main_concern: str = ""
    advice: str = ""
    caregiver_alert: str = ""
    next_step: str = ""
    known_facts: list[str] = Field(default_factory=list)
    unknowns: list[str] = Field(default_factory=list)
    follow_up_questions: list[str] = Field(default_factory=list)
    raw: str = ""


class ChatResponse(BaseModel):
    final_message: str
    active_mode: str = "Mock"
    final_message_source: str = "template"
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
    activated_agents: list[str]
    overall_risk: RiskLevel
    main_concern: str
    advice: str
    caregiver_alert: str
    next_step: str
    memory_id: int
    human_confirmation_required: bool = False
    confirmation_reason: str = ""
    routing_explanation: str = ""
    graph_reasoning: dict[str, Any] = Field(default_factory=dict)
    detected_signals: list[dict[str, str]] = Field(default_factory=list)
    risk_scores: dict[str, int] = Field(default_factory=dict)
    memory_summary: dict[str, Any] = Field(default_factory=dict)
    situation: dict[str, Any] = Field(default_factory=dict)
    care_plan: dict[str, Any] = Field(default_factory=dict)
    developer_state: dict[str, Any] = Field(default_factory=dict)


class LogRecord(BaseModel):
    id: int
    timestamp: str
    user_id: str
    message: str
    final_message: str = ""
    activated_agents: list[str]
    overall_risk: str
    main_concern: str
    advice: str
    caregiver_alert: str
    next_step: str
    raw_json: dict[str, Any]


class KaggleDownloadResponse(BaseModel):
    downloaded: list[str]
    cached: list[str]
    errors: list[str]
    csv_files: list[str]
