# Interview Questions

## Why is this a multi-agent system?

The user sees one companion, but the backend separates responsibilities into routing, specialist reasoning, alert policy, conversation response, guardrails, and memory. Each part has a clear role and produces inspectable state.

## Why use a hybrid router?

Safety-sensitive messages need deterministic handling, while ordinary messages can benefit from classifier help. The hybrid router lets rules override the ML router for high-risk or multi-topic cases.

## Why Naive Bayes?

Naive Bayes is lightweight, explainable, fast, and suitable for a small prototype classifier. It is not treated as a final authority; it supports routing when rule evidence is weak.

## Why rule-based override?

For eldercare safety, missing obvious risk is worse than choosing a less elegant path. Rule override makes critical cases such as dizziness, medication uncertainty, or fraud pressure auditable.

## Why use RAG?

RAG lets the system retrieve local care knowledge instead of relying only on prompt memory. In this prototype it supports safer, more consistent context without external knowledge APIs.

## Why use GraphRAG?

GraphRAG expresses relationships like `dizziness -> fall risk -> sit down -> red-flag questions`. That makes routing and recommendations easier to explain than keyword matching alone.

## What types of memory exist?

The project distinguishes conversation turns, interaction logs, care events, user summaries, confirmed profile facts, and do-not-save data. The memory classifier decides what should be stored.

## How do guardrails work?

Guardrails run after the conversation agent. They check for unsafe claims, alert mismatches, hidden traceback exposure, and actions the prototype cannot actually perform.

## Where is human approval placed?

Before real-world action. Alerts, reminders, and future navigation actions are prepared only. A caregiver or user must approve any future external integration.

## Is this reinforcement learning?

No. The prototype uses rules, local classification, memory, RAG/GraphRAG, and an optional LLM response writer. It does not learn through reward feedback or policy optimization.

## How do you evaluate the system?

Use scenario tests, routing checks, alert policy checks, guardrail checks, role/access checks, and human review of developer traces. The README and evaluation plan avoid claiming production metrics.

## How would you add tool calling?

Start with mock tools that return structured objects. Then add real integrations only behind permission, consent, audit logs, error handling, and human confirmation.

## How would you make it production-ready?

Add proper identity, consent, audit logging, clinical review, data retention controls, monitoring, rate limits, formal evaluation, deployment security, and carefully approved external integrations.
