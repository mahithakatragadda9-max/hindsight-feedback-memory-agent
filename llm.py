import json
from typing import Any
from groq import Groq
from config import GROQ_API_KEY, GROQ_MODEL
from models import FeedbackAnalysis


class GroqUnavailable(Exception):
    pass


_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if not GROQ_API_KEY:
        raise GroqUnavailable("GROQ_API_KEY is not configured.")
    if _client is None:
        _client = Groq(api_key=GROQ_API_KEY)
    return _client


ANALYSIS_SYSTEM = """You are a product feedback understanding engine.
Extract structured information from a single piece of customer feedback.
Rules:
- Do not invent facts not present in the feedback.
- Do not use industry-specific hard-coded assumptions.
- Generate the theme from the feedback itself (short noun phrase).
- Return strict JSON only, matching the requested keys exactly."""

ANALYSIS_SCHEMA = {
    "summary": "one-sentence summary of what the user said",
    "underlying_issue": "the actual product/user problem in neutral wording",
    "theme": "short dynamically discovered theme (2-4 words, Title Case)",
    "sentiment": "positive | neutral | negative | mixed",
    "priority": "low | medium | high | critical",
    "affected_capability": "the product capability / feature area",
    "requested_need": "what the user wants or expects",
    "recommended_action": "immediate product action (one sentence)",
    "failure_mode": "what specifically fails, if anything, in neutral wording",
    "confidence": 0.0
}


def analyze_feedback(text: str, workspace: str, channel: str) -> FeedbackAnalysis:
    client = _get_client()
    prompt = f"""Workspace: {workspace}
Channel: {channel}
Feedback:
\"\"\"{text}\"\"\"

Return JSON with exactly these keys:
{json.dumps(ANALYSIS_SCHEMA, indent=2)}"""

    try:
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": ANALYSIS_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
        )
    except Exception as e:
        raise GroqUnavailable(f"Groq analysis failed: {e}") from e

    raw = resp.choices[0].message.content or "{}"
    data = json.loads(raw)

    clean: dict[str, Any] = {}
    for key in ANALYSIS_SCHEMA:
        clean[key] = data.get(key, ANALYSIS_SCHEMA[key] if key != "confidence" else 0.0)
    try:
        clean["confidence"] = float(clean.get("confidence", 0.0))
    except (TypeError, ValueError):
        clean["confidence"] = 0.0
    return FeedbackAnalysis(**clean)


EVIDENCE_SYSTEM = """You are an evidence gate for a feedback memory agent.

You will receive:
1. NEW feedback analysis
2. HISTORICAL memories
3. PRODUCT changes

For each historical memory, decide whether it describes the SAME underlying
customer problem as the new feedback.

IMPORTANT:
- Treat paraphrases and minor wording changes as the SAME issue when the
  underlying product problem and expected behavior are the same.
- Do NOT require exact wording or exact keywords.
- For example:
  "My payment failed while sending money"
  and
  "My payment failed while sending money again"
  describe the SAME underlying issue.
- Shared keywords alone are NOT sufficient.
- Compare underlying_issue, failure_mode, affected_capability, requested need,
  and expected behavior.
- The word "again", "still", "once more", "repeatedly", or similar recurrence
  language strengthens the evidence that the same issue has returned.
- Set same_issue=true when the historical experience and new feedback clearly
  describe the same underlying problem, even if the wording differs.
- Set relevance between 0.0 and 1.0:
    0.90-1.00 = essentially the same issue
    0.75-0.89 = strongly related underlying issue
    0.50-0.74 = related but insufficiently specific
    below 0.50 = weak/unrelated
- Do not inflate relevance merely because the same product area is mentioned.
- If genuinely uncertain, set same_issue=false.

Also determine whether any PRODUCT CHANGE addresses the SAME underlying issue.

Return strict JSON only:

{
  "evaluations": [
    {
      "memory_id": "...",
      "relevance": 0.0,
      "same_issue": true,
      "reason": "brief evidence-based explanation"
    }
  ],
  "matched_product_change_id": "..." | null
}
"""

def evaluate_evidence_llm(new_analysis: dict, memories: list[dict], product_changes: list[dict]) -> dict:
    client = _get_client()
    if not memories and not product_changes:
        return {"evaluations": [], "matched_product_change_id": None}

    prompt = f"""NEW FEEDBACK ANALYSIS:
{json.dumps(new_analysis, indent=2)}

HISTORICAL MEMORIES:
{json.dumps(memories, indent=2)}

PRODUCT CHANGES:
{json.dumps(product_changes, indent=2)}

Return JSON as instructed."""

    try:
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": EVIDENCE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
    except Exception as e:
        raise GroqUnavailable(f"Groq evidence evaluation failed: {e}") from e

    raw = resp.choices[0].message.content or "{}"
    data = json.loads(raw)
    if "evaluations" not in data:
        data["evaluations"] = []
    if "matched_product_change_id" not in data:
        data["matched_product_change_id"] = None
    return data
