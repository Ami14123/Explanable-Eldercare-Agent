# ElderGuard Baseline Audit

## Date

YYYY-MM-DD

## Git branch

upgrade/portfolio-v2

## Environment

- Operating system: Windows
- Python version:3.12.10
- LLM mode: mock
- FastAPI URL: http://127.0.0.1:8000
- Streamlit URL: http://localhost:8501

## Application startup

| Component | Result | Notes |
|---|---|---|
| Dependency installation | Pass / Fail | |
| FastAPI startup | Pass / Fail | |
| `/status` endpoint | Pass / Fail | |
| Swagger documentation | Pass / Fail | |
| Streamlit startup | Pass / Fail | |
| SQLite connection | Pass / Fail | |
| RAG store loading | Pass / Fail | |

## Existing tests

| Test | Result | Notes |
|---|---|---|
| Reasoning workflow | Pass / Fail | |
| Alert tests | Pass / Fail | |
| Guardrail tests | Pass / Fail | |
| Memory classifier | Pass / Fail | |
| LLM mode tests | Pass / Fail | |
| Example tests | Pass / Fail | |
| Backend agent tests | Pass / Fail | |

## Manual chat cases

| Input | Selected topic | Risk | Alert | Quality notes |
|---|---|---|---|---|
| Lonely user | | | | |
| Missed medicine | | | | |
| Bank password request | | | | |
| Dizziness | | | | |
| Hunger and juice | | | | |

## Known problems

1.
2.
3.

## Baseline conclusion

The current system is working / partially working / not working.

The next upgrade phase will focus on security and privacy without changing the intended chatbot behavior.