# ElderGuard AI

Autonomous Multi-Agent ElderCare Companion with a FastAPI backend, simplified LangGraph workflow, SQLite memory, optional Kaggle dataset access, local care-graph retrieval, OpenRouter live LLM support, and a polished Streamlit local demo UI.

Architecture:

```text
Streamlit UI or Dify UI
-> HTTP request
-> FastAPI /chat endpoint
-> LangGraph StateGraph workflow
-> memory_context_builder
-> router
-> broad_agent_reasoning
   -> Safety Agent
   -> Health & Daily Care Agent
   -> Emotional & Social Agent
   -> Action Agent
-> alert_decision
-> conversation_agent
-> guardrails
-> save_memory
-> response back to the UI
```

The app works in deterministic mock mode when `LLM_MODE=mock`. Set `LLM_MODE=live` with `OPENROUTER_API_KEY` and `OPENROUTER_MODEL` to use OpenRouter. OpenRouter is the only live LLM provider.

The user interacts with one general chatbot. The app is still a multi-agent system, but it is simplified into four broad local agents. Only the `conversation_agent` calls the LLM. Router, broad agents, GraphRAG context, guardrails, memory, and traces are local Python logic. Target: `llm_calls_this_turn = 1`.

Advanced prototype concepts included:

- Harness + Loop + LLM Ops pattern through LangGraph
- `memory_context_builder -> router -> broad_agent_reasoning -> conversation_agent -> guardrails -> save_memory`
- Four broad local agents: Safety, Health & Daily Care, Emotional & Social, Action
- Local Alert Decision node for caregiver, health, fraud, and emergency alerts
- One LLM call per chat turn in normal provider mode
- Shared OpenRouter gateway for all live LLM calls
- Trace/eval artifacts in `data/traces/last_trace.json`
- Real Conversation Agent error logs in `data/conversation_agent_error.txt`
- Lightweight care knowledge graph in `app/care_graph.py`
- Agentic memory timeline from SQLite logs
- Conversation-turn memory with `conversation_id` so follow-up replies stay in context
- Human-in-the-loop safety flags for high-risk cases
- Explainable routing with detected signals, router decision, active topic, activated agents, and LLM call count
- Three-view Streamlit interface: Elder View, Family Summary, and Technical Trace

## Setup

Windows commands:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
streamlit run streamlit_app.py
```

Create a local `.env` file from `.env.example`:

```powershell
copy .env.example .env
```

Run the backend:

```powershell
uvicorn app.main:app --reload
```

Run the Streamlit demo:

```powershell
streamlit run streamlit_app.py
```

Keep the FastAPI terminal open while using Streamlit. If you need another command, open a second terminal.

## Environment Variables

```env
OPENROUTER_API_KEY=
OPENROUTER_MODEL=openai/gpt-4o-mini
LLM_MODE=mock
KAGGLE_USERNAME=
KAGGLE_KEY=
KAGGLE_API_TOKEN=
```

## OpenRouter API Key

OpenRouter is the recommended provider for this prototype.

```env
LLM_MODE=live
OPENROUTER_API_KEY=your_openrouter_key_here
OPENROUTER_MODEL=openai/gpt-4o-mini
```

Then restart FastAPI:

```powershell
uvicorn app.main:app --reload
```

Developer Console should show `llm_mode=live`, `llm_provider=openrouter`, `llm_model=openai/gpt-4o-mini`, and normal successful turns should show `llm_calls_this_turn=1`.

## Kaggle API Key

Kaggle datasets are optional. The app skips Kaggle downloads when credentials are missing.

New Kaggle token style:

```env
KAGGLE_API_TOKEN=your_kaggle_api_token_here
```

This uses `kagglehub`, so run `pip install -r requirements.txt` after adding the token.

Old Kaggle username/key style:

```env
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_key
```

To enable Kaggle:

1. Create a Kaggle API token from your Kaggle account settings.
2. Open `.env` and set:

```env
KAGGLE_API_TOKEN=your_kaggle_api_token_here
```

3. Restart FastAPI.
4. Download datasets from Streamlit using **Download Kaggle datasets**, or call:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/download-kaggle-datasets
```

Downloaded files are cached in `data/raw/`. Existing files are not downloaded again.

## Local RAG Knowledge

Sample knowledge files live in:

```text
data/knowledge/
```

Included files:

- `eldercare_safety.md`
- `medication_safety.md`
- `scam_prevention.md`
- `fall_prevention.md`

The RAG layer uses LangChain document loading and a local FAISS vector store. It does not require any paid embedding API.

Rebuild the vector store from Streamlit using **Rebuild RAG vector store**, or run:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/rebuild-vector-store
```

You can also rebuild from Python:

```powershell
python -c "from app.rag import build_vector_store; print(build_vector_store())"
```

If no vector store exists, the app still runs. The agents simply answer without retrieved context until the store is built.

## Graph RAG Retriever

ElderGuard also uses a lightweight care knowledge graph in `app/care_graph.py`. This is separate from the FAISS document store and is used inside the LangGraph workflow before orchestration.

Graph RAG returns:

- detected nodes
- reasoning path
- still missing information
- recommended next question
- relevant safety rules
- recommended next action

Example paths:

- `hunger -> basic_need -> check_available_food -> if_only_juice_suggest_simple_food`
- `dizziness -> fall_risk -> sit_or_lie_down -> caregiver_alert_if_worsening`
- `unknown_email -> link_risk -> do_not_click -> verify_sender`
- `missed_medication -> dosage_confusion -> check_label -> contact_pharmacist`
- `loneliness -> emotional_distress -> contact_friend -> calming_activity`

## API

### POST `/chat`

Request:

```json
{
  "message": "I forgot my blood pressure medicine and now I feel dizzy.",
  "user_id": "demo_user",
  "conversation_id": "default"
}
```

`conversation_id` is optional. If it is missing, ElderGuard uses `"default"`. Use the same `conversation_id` for each message in the same chat so ElderGuard can understand follow-up answers.

Response:

```json
{
  "final_message": "A warm situation-aware response for the user...",
  "current_topic": "health_symptom",
  "is_follow_up": false,
  "known_information": [],
  "follow_up_questions": ["Do you have chest pain or trouble breathing?"],
  "specificity_score": 90,
  "missing_information": ["whether the user is alone"],
  "hypotheses": [],
  "need_more_information": true,
  "reasoning_summary": "Situation analyzed as health or fall safety...",
  "activated_agents": ["health_agent", "medication_agent"],
  "overall_risk": "high",
  "main_concern": "...",
  "advice": "...",
  "caregiver_alert": "...",
  "next_step": "...",
  "memory_id": 1,
  "human_confirmation_required": true,
  "confirmation_reason": "High-risk case...",
  "routing_explanation": "...",
  "graph_reasoning": {},
  "detected_signals": [],
  "risk_scores": {},
  "memory_summary": {}
}
```

### GET `/logs`

Returns recent SQLite interaction logs.

### GET `/status`

Returns backend status, LLM mode, SQLite status, and Kaggle cache status for the Streamlit sidebar.

### POST `/download-kaggle-datasets`

Downloads and caches optional Kaggle datasets into `data/raw/`.

Default datasets:

- `uciml/sms-spam-collection-dataset`
- `ziya07/elderly-fall-detection-iot-dataset`
- `programmer3/iot-based-chronic-medication-adherence-dataset`

### POST `/rebuild-vector-store`

Builds or rebuilds the local FAISS vector store from `data/knowledge/` and any cached Kaggle CSV files in `data/raw/`.

## Dify HTTP Request Node

Method: `POST`

URL:

```text
http://localhost:8000/chat
```

Headers:

```text
Content-Type: application/json
```

Body:

```json
{
  "message": "{{sys.query}}",
  "user_id": "demo_user",
  "conversation_id": "default"
}
```

Dify should use the response JSON to answer the user.

## Streamlit Demo Inputs

Try these in the local UI:

- Daily need: `I want juice but I don't have money to buy`
- Unknown email: `Someone emailed me but I don't know them, should I click the link?`
- Dizzy after standing: `I feel dizzy after standing up`
- Missed medicine: `I forgot my blood pressure medicine`
- Lonely: `I feel lonely because my friends are busy`
- Dizzy at home: `I feel dizzy and weak after almost falling in the kitchen.`
- Missed medicine: `I forgot whether I took my blood pressure medicine this morning.`
- Suspicious call: `A bank caller asked me for my OTP and told me to transfer money urgently.`
- Feeling lonely: `I feel lonely today and I just want someone to talk to.`
- Mixed concerns: `I feel dizzy and weak after almost falling in the kitchen, someone called me to ask for my bank account, and I need my friend support.`

## App Views

### Elder View

The default experience is a calm, simple chatbot for older adults. It shows:

- ElderGuard title
- A welcoming sentence
- Conversation history
- Quick suggestion buttons
- One chat input
- Caregiver alert only when needed

The Elder View does not show raw JSON, backend URLs, model calls, Graph RAG, prompts, internal agents, or developer traces.

### Family Summary

Family Summary is a separate nontechnical view for authorized caregivers or family members. It explains:

- What the elder said
- What the system detected
- Risk level
- Broad support area
- Why the response was given
- Information used from local care knowledge
- Whether caregiver action was prepared
- Suggested next step

This view is deterministic. It is built from structured backend fields and does not call an LLM.

### Technical Trace

Technical Trace is the observability view for developers. It shows:

- Total duration
- Selected specialist
- Risk level
- OpenRouter model
- LLM calls this turn
- RAG sources
- Alert status
- Guardrail status
- LangGraph spans in execution order
- Output guardrail before and after values
- Sanitized advanced technical data

Sensitive values such as API keys, authorization headers, hidden prompts, and secrets are redacted before display.

The user does not choose agents in any elder-facing workflow. General reasoning, broad agents, Graph RAG, alert policy, guardrails, and memory run silently in the backend.

## Conversation Memory

ElderGuard stores conversation turns in SQLite using:

- `user_id`
- `conversation_id`
- `role`
- `message`
- `timestamp`

Before each response, the backend loads the recent conversation and checks whether the new message is a follow-up answer. For example:

```text
User: Someone emailed me but I don't know them, should I click the link?
ElderGuard: Who sent it, and were you expecting it?
User: my daughter's friend
```

ElderGuard keeps the topic as email safety and continues with the fraud/safety agent instead of treating `"my daughter's friend"` as a new support request.

### Privacy Model

Advanced AI components are hidden from elderly users for trust and simplicity. Family members see a concise care explanation, not chain of thought or secrets. Developers can inspect sanitized traces in the Technical Trace view.

## Screenshots

Add screenshots here for your report or presentation:

- Main chatbot dashboard
- ElderGuard response card
- Behind-the-scenes activated agents
- Recent memory logs

## Test Examples

Run the conversation-state reasoning tests:

```powershell
python run_reasoning_workflow_tests.py
```

These tests verify the redesigned logic before any final answer is written:

- topic switch: fraud to food
- topic switch: food to medication
- yes/no after a pending question
- user correction
- user asks for a message draft
- normal general question
- repeated advice tracking
- impossible action alternative
- multi-agent mixed issue
- new unrelated topic after a high-risk topic

Run the automatic backend workflow test while FastAPI is running:

```powershell
python run_agent_tests.py
```

This calls `POST /chat` like the Streamlit app does. It checks:

- conversation memory across multiple turns
- topic switching without resetting the chat
- correct specialist agent activation
- health-only, medication-only, fraud-only, companion-only, and basic-needs behavior
- no unrelated safety advice in the final answer

Run the built-in examples:

```powershell
python test_examples.py
```

Covered cases:

- Hunger plus juice follow-up
- Suspicious email plus sender follow-up
- Normal daily-life question
- Lonely because family is busy
- Dizziness after standing
- Forgotten medication
- Juice but no money to buy
- Unknown email link
- Dizziness after standing up
- Forgot blood pressure medicine
- Lonely because friends are busy
- Suspicious email follow-up
- Medication follow-up
- Health symptom follow-up
- Loneliness follow-up

Expected behavior:

- Agents stay hidden from the user.
- Only relevant agents activate.
- Final answer starts with empathy or acknowledgement.
- Final answer uses the exact user situation.
- Follow-up questions appear when details are missing.
- Short follow-up answers continue the previous topic.
- No unrelated medication, fraud, health, or loneliness advice is added.
