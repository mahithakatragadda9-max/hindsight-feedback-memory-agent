# Feedback Memory Intelligence Agent

A Streamlit hackathon prototype for an agent that treats customer feedback
as an evolving experience lifecycle, not a static dataset.

Core loop:

    FEEDBACK → EXPERIENCE → MEMORY → PRODUCT DECISION
             → PRODUCT CHANGE → OUTCOME → FUTURE FEEDBACK

Hindsight is used as persistent experience memory. The agent recalls prior
experiences, evaluates whether they are genuinely relevant, classifies the
new feedback into a lifecycle state (NOVEL / RECURRING / IMPROVING_RESOLVED /
POSSIBLE_REGRESSION / INSUFFICIENT_EVIDENCE), and retains the new experience.

## Setup

    python -m venv .venv
    source .venv/bin/activate       # Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    cp .env.example .env            # fill in your keys

Required `.env` values:

    GROQ_API_KEY=...
    GROQ_MODEL=llama-3.3-70b-versatile
    HINDSIGHT_API_URL=https://...   # base URL, no trailing slash
    HINDSIGHT_API_KEY=...
    HINDSIGHT_BANK_ID=...

## Run

    python selftest.py        # offline logic test, no keys needed
    streamlit run app.py

On first run, `data/seed_feedback.jsonl` (60 synthetic demo records) and
`data/product_changes.jsonl` (a demo change) are created if missing.

## Demo flow

1. Open the app — see 60 synthetic seed records.
2. Go to **🧪 Judge Test Mode** — upload any CSV/JSON/JSONL, or paste records.
   The agent analyzes each record with Groq and retains experiences in Hindsight.
3. Go to **🎤 Live Feedback** — submit a brand-new feedback item.
   - First time a theme appears → NOVEL
   - Paraphrase of a prior item → RECURRING
4. Record a product change for the theme.
5. Submit a recurrence after the change → POSSIBLE_REGRESSION (only if
   the improvement signal and pre/post-change evidence chain is complete).
6. Open **🕰️ Feedback Time Machine** to see the chain visually.

## Evidence gating

Memories only influence the lifecycle decision if a second LLM call
(`evaluate_evidence_llm`) marks them as `same_issue=true` AND relevance ≥ 0.75.
Shared keywords alone are rejected.

`POSSIBLE_REGRESSION` requires ALL of:

1. Strongly related pre-change experience
2. A recorded product change for that theme
3. A post-change improvement signal (same-theme volume drops)
4. A strongly related post-change recurrence

If any condition is missing, the lifecycle is `INSUFFICIENT_EVIDENCE` or
`RECURRING` — never a regression.

## Storage

- `data/seed_feedback.jsonl` — immutable synthetic demo history (never modified)
- `data/live_feedback.jsonl` — appended for every import / live / voice submission
- `data/product_changes.jsonl` — product change records

Total count = seed + live. It is computed from the files, never hard-coded.

## Hindsight adapter

`hindsight_client.py` uses this contract:

    POST {HINDSIGHT_API_URL}/banks/{bank_id}/memories   (retain)
    POST {HINDSIGHT_API_URL}/banks/{bank_id}/recall     (recall)

If your Hindsight instance uses different paths, only this file needs updating.

When Hindsight fails or is not configured, the app still analyzes feedback
and shows a clear warning — it never claims memory was used.

## Known limitations

- The Hindsight endpoint shape in `hindsight_client.py` assumes a bank-based
  REST contract. Adjust the two path strings if your deployment differs.
- The lifecycle engine uses a simple volume-based improvement signal; more
  sophisticated regression detection would need outcome telemetry.
- Voice transcription requires a Groq key and a valid audio file.
