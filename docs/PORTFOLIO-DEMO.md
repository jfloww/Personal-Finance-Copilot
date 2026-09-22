# MyFinSecretary: two-minute portfolio walkthrough

**Who it helps:** a finance-operations analyst investigating unusual software charges.
**Question:** “Why did August spend increase, and which charge needs review?”

This is a public, **synthetic-data** backend showcase. The web page runs a scripted,
deterministic investigation, **not** a live model. It accepts no uploads or arbitrary prompts.

## Show it

1. Run `cd backend && uv run uvicorn --app-dir src offerdelta.api.main:app --reload`.
2. Open `http://127.0.0.1:8000/`. Sample A reconciles a $12,500 increase: a
   $7,650 renewal and a $4,850 possible duplicate. Open the tool trace and policy citation.
3. Click **Sample B · Billing review**. It parses a committed synthetic CSV and
   computes a different $5,500 increase ($3,500 duplicate candidate, $2,000 renewal).
   The merchant, transaction IDs, spend drivers, and trace change together.
4. Show `POST /v1/demo/agent/run` in `/docs`. The request permits only the two
   bundled scenarios. The response uses exact decimal strings and returns a proposal,
   never a ledger or review-queue write.

## What the implementation proves

- Six schema-validated, read-only tools are shared by the in-process runtime and an
  MCP stdio server. A contract test compares the MCP-advertised schemas and results
  with direct registry calls.
- A bounded agent runtime and optional local model path exist, but the web demo does
  not invoke them. This distinction is visible on the page and in the API response.
- Financial deltas and merchant drivers use `Decimal`; tests assert that their sum
  reconciles. Duplicate matches are **candidates**, not proof of a double payment.
- The public policy excerpt is versioned and cited, but lookup is keyword-based;
  it is **not RAG over real policies**.
- A review proposal is deliberately unpersisted. Human approval and any subsequent
  transaction change are **not** implemented in this public workflow.

## Source map

| Concern | File |
| --- | --- |
| Investigation orchestration | `backend/src/offerdelta/application/queries/operations_demo.py` |
| Tool schemas and implementations | `backend/src/offerdelta/agent/tools/operations.py` |
| Synthetic CSV parser and allowlisted case | `backend/src/offerdelta/demo/operations_cases.py` |
| API contract | `backend/src/offerdelta/api/main.py` |
| Browser workbench | `backend/src/offerdelta/api/static/agent.html` |
| Behavioral and MCP tests | `backend/tests/unit/agent/test_operations.py`, `backend/tests/contract/` |

## Before sharing externally

- Commit and push the current agent work; untracked files are invisible on GitHub.
- Confirm CI with PostgreSQL and check the deployed `/` page is this version.
- Record a short screen capture of both scenarios and the trace. Link it from the
  README so a reviewer can understand the project without local setup.
- Do not quote the historical categorisation benchmark as an operations-agent score.
  A separate held-out investigation evaluation is needed before that claim.
