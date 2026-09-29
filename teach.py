"""Teach / Correction: user-taught theme rules, persisted per workspace.

A rule says: "feedback containing any of these phrases belongs to THIS theme".
Rules are applied deterministically (no LLM) right after analysis, so they
change what future feedback is classified as. Each rule is also retained in
Hindsight through the existing HindsightClient.retain() so the knowledge
lives in the workspace's memory too.
"""
from __future__ import annotations

import re
import uuid
from datetime import datetime
from typing import Optional

from models import Memory
from storage import DATA_DIR, append_jsonl, load_jsonl

TEACH_FILE = DATA_DIR / "teach_rules.jsonl"

# Used to recognise (and keep out of lifecycle evidence) Hindsight memories
# that are taught rules rather than customer feedback.
TEACH_CHANNEL = "Teaching"
TEACH_MARKER = "[Teaching rule]"

_STOP = {"a", "an", "the", "to", "of", "is", "in", "on", "for", "and", "or"}


# ---------------------------------------------------------------- parsing
def parse_phrases(raw: str) -> list[str]:
    """Split 'a, b\\nc; d' into unique, non-empty phrases (case-insensitive dedupe)."""
    out: list[str] = []
    seen: set[str] = set()
    for part in re.split(r"[,\n;]+", raw or ""):
        p = part.strip().strip("\"'“”‘’").strip()
        key = _norm(p)
        if key and key not in seen:
            seen.add(key)
            out.append(p)
    return out


def _norm(text: str) -> str:
    return re.sub(r"[\W_]+", " ", str(text or "").lower()).strip()


def _phrase_matches(phrase_norm: str, text_norm: str) -> bool:
    if not phrase_norm or not text_norm:
        return False
    padded = f" {text_norm} "
    if f" {phrase_norm} " in padded:  # whole-word phrase match
        return True
    tokens = [t for t in phrase_norm.split() if t not in _STOP]
    # multi-word phrase: all words present, any order ("export the csv")
    return len(tokens) >= 2 and all(f" {t} " in padded for t in tokens)


# ---------------------------------------------------------------- storage
def load_rules(workspace: str) -> list[dict]:
    ws = (workspace or "").strip()
    return [r for r in load_jsonl(TEACH_FILE) if str(r.get("workspace", "")).strip() == ws]


def find_matching_rule(text: str, rules: list[dict]) -> Optional[dict]:
    """Best rule for this text: longest matched phrase wins, then newest rule."""
    t = _norm(text)
    best: Optional[tuple] = None
    best_hit: Optional[dict] = None
    for rule in rules:
        theme = str(rule.get("theme") or "").strip()
        if not theme:
            continue
        for phrase in rule.get("phrases") or []:
            p = _norm(phrase)
            if _phrase_matches(p, t):
                score = (len(p), str(rule.get("created_at") or ""))
                if best is None or score > best:
                    best = score
                    best_hit = {"rule_id": rule.get("rule_id"), "theme": theme, "phrase": phrase}
    return best_hit


def apply_taught_theme(analysis, text: str, workspace: str,
                       rules: Optional[list[dict]] = None) -> Optional[dict]:
    """If a taught rule matches `text`, overwrite analysis.theme. Never raises.

    Returns the matched-rule info dict, or None when nothing matched.
    """
    if analysis is None:
        return None
    try:
        hit = find_matching_rule(text, load_rules(workspace) if rules is None else rules)
    except Exception:
        return None
    if hit:
        analysis.theme = hit["theme"]
    return hit


def is_teaching_memory(memory: Memory) -> bool:
    """True for recalled Hindsight memories that are taught rules, not feedback."""
    return (
        str(getattr(memory, "channel", "")).strip() == TEACH_CHANNEL
        or TEACH_MARKER in str(getattr(memory, "raw_text", ""))
    )


# ------------------------------------------------------- teach + retain
def _retain_in_hindsight(rule: dict, client=None) -> tuple[bool, str]:
    """Store the rule with the existing HindsightClient.retain(). Best effort."""
    try:
        from hindsight_client import HindsightClient, HindsightUnavailable
    except Exception as e:  # noqa: BLE001
        return False, f"Hindsight client could not be loaded ({e})."
    client = client or HindsightClient()
    phrases = "; ".join(rule["phrases"])
    mem = Memory(
        memory_id=rule["rule_id"],
        feedback_id=f"teach:{rule['rule_id']}",
        workspace=rule["workspace"],
        timestamp=rule["created_at"],
        channel=TEACH_CHANNEL,
        raw_text=(
            f"{TEACH_MARKER} Feedback mentioning any of: {phrases} "
            f"must be classified under the theme '{rule['theme']}'."
        ),
        discovered_theme=rule["theme"],
        underlying_issue=f"User-taught classification rule for '{rule['theme']}'",
        sentiment="neutral",
        priority="medium",
        agent_interpretation="Correction taught by the user; apply to future feedback in this workspace.",
        lifecycle_state="TAUGHT_RULE",
    )
    try:
        client.retain(mem)
        return True, "Retained in Hindsight."
    except HindsightUnavailable as e:
        return False, f"Saved locally, but Hindsight retain was skipped: {e}"
    except Exception as e:  # noqa: BLE001
        return False, f"Saved locally, but Hindsight retain failed: {e}"


def teach(workspace: str, theme: str, phrases_raw: str, client=None) -> dict:
    """Validate, retain in Hindsight, and persist a correction for `workspace`.

    Returns {"ok": bool, "error"?, "rule"?, "hindsight_ok"?, "hindsight_message"?}.
    """
    workspace = (workspace or "").strip()
    theme = re.sub(r"\s+", " ", (theme or "")).strip()
    phrases = parse_phrases(phrases_raw)
    if not workspace:
        return {"ok": False, "error": "No workspace selected."}
    if not theme:
        return {"ok": False, "error": "Enter the theme the agent should use."}
    if len(theme) > 80:
        return {"ok": False, "error": "Theme name is too long (max 80 characters)."}
    if not phrases:
        return {"ok": False, "error": "Enter at least one phrase (separate several with commas)."}

    rule = {
        "rule_id": str(uuid.uuid4()),
        "workspace": workspace,
        "theme": theme,
        "phrases": phrases,
        "created_at": datetime.utcnow().isoformat(timespec="seconds"),
    }
    h_ok, h_msg = _retain_in_hindsight(rule, client)
    rule["hindsight_retained"] = h_ok
    append_jsonl(TEACH_FILE, rule)
    return {"ok": True, "rule": rule, "hindsight_ok": h_ok, "hindsight_message": h_msg}