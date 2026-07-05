# ElderGuard Framework Mapping

ElderGuard is designed as a controlled multi-agent loop for an interview portfolio. The goal is not to show a production healthcare platform, but to show how a sensitive AI assistant can be structured, observed, and constrained.

## Harness

The harness is the outer system that makes the agent safe to run locally:

- Streamlit provides the elder-facing chat, caregiver view, and developer trace.
- FastAPI exposes the `/chat` gateway and protected admin endpoints.
- Pydantic schemas define request and response contracts.
- The gateway validates message shape, length, user id, conversation id, and request id.
- Role separation keeps family and developer views behind environment-configured access codes.

## Loop

The loop is the LangGraph workflow:

```text
memory_context_builder
-> router
-> broad_agent_reasoning
-> alert_decision
-> conversation_agent
-> guardrails
-> save_memory
```

Each node adds structured state. The elder user only sees the final natural response.

## Router

The router combines deterministic rules and optional local ML routing. Safety rules win when risk is clear, because missed safety signals are more harmful than overly clever classification.

The router returns:

- active topic
- activated agents
- selected agent
- detected signals
- priority
- explanation data

## Specialist Agents

The current prototype uses four broad specialist areas:

- Safety Agent: fraud, suspicious links, unsafe pressure, urgent safety signals.
- Health & Daily Care Agent: symptoms, medication uncertainty, food, mobility, daily needs.
- Emotional & Social Agent: loneliness, sadness, family support, reassurance.
- Action Agent: drafts messages, checklists, reminders, caregiver notes, and call scripts.

Specialists create structured evidence. They do not directly speak to the elder user.

## Q&A / Conversation Agent

The conversation agent is the final speaker. It receives the latest message, recent history, router decision, GraphRAG context, specialist evidence, alert decision, and previous assistant response.

Its job is to produce one warm, simple, situation-aware answer.

## Memory

SQLite supports local memory:

- Recent conversation turns for continuity.
- Interaction logs for admin review.
- Care events classified by memory type.
- User summaries and confirmed profile facts.

Memory should inform context, not override the latest user message.

## RAG

Local RAG retrieves short snippets from `data/knowledge/` and optional cached demo datasets. It is used as background context. It does not replace professional medical guidance.

## GraphRAG

GraphRAG is a lightweight care graph implemented in Python. It maps user signals to explainable care pathways such as:

- dizziness to fall risk and red-flag questions
- unknown email to link verification
- missed medication to dosage uncertainty and pharmacist confirmation
- hunger to available food checking

## Guardrails

Guardrails check that responses do not diagnose, prescribe, claim alerts were sent, or expose secrets. They also record issues for the developer trace.

## Human Approval

Human approval is placed before real-world action:

- Alerts are prepared, not sent.
- Reminders are prepared, not scheduled.
- Navigation is a mock roadmap feature and requires permission before any future real route integration.

## Observability

The Developer Trace page and `data/traces/last_trace.json` show sanitized workflow data:

- LangGraph path
- router decision
- activated agents
- alert decision
- LLM mode and call count
- guardrail status

## Evaluation

Evaluation is prototype-focused:

- Does the correct specialist activate?
- Does the final answer avoid unrelated warnings?
- Does an alert trigger only when policy requires it?
- Are logs/admin endpoints protected?
- Are secrets and tracebacks hidden from public responses?
