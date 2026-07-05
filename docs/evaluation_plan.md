# ElderGuard Evaluation Plan

This file lists planned evaluation metrics. It does not report measured production results.

## Technical Metrics

- Routing accuracy on synthetic scenarios.
- Correct activated agent set.
- LLM calls per chat turn.
- Backend latency.
- Streamlit-to-FastAPI success rate.
- Test pass rate.
- Error handling quality.
- Trace completeness.

## Business/Social Value Metrics

- Whether the elder receives one clear next step.
- Whether the answer is understandable to a nontechnical user.
- Whether caregiver summaries are useful.
- Whether the system reduces repeated clarification.
- Scenario coverage across safety, health, medication, emotional, and daily needs.

## Risk/Trust Metrics

- Missed alert rate on synthetic high-risk scenarios.
- Unnecessary alert rate on low-risk scenarios.
- False medical certainty.
- Medication dosage-change violations.
- Claiming alerts were sent when they were only prepared.
- Privacy leakage into public UI.
- Traceback or secret exposure.

## Operational Metrics

- Local setup time.
- Required environment variables are documented.
- Admin endpoints reject missing or wrong tokens.
- `.env` and local databases remain uncommitted.
- Kaggle and RAG operations fail safely when optional data is unavailable.
- Developer trace remains readable for debugging.
