import json
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Iterable

DATA_DIR = Path("data")
SEED_FILE = DATA_DIR / "seed_feedback.jsonl"
LIVE_FILE = DATA_DIR / "live_feedback.jsonl"
CHANGES_FILE = DATA_DIR / "product_changes.jsonl"
WORKSPACES_FILE = DATA_DIR / "workspaces.jsonl"

# Built-in workspaces keep workspace_id == display name (backward compatible).
BUILTIN_WORKSPACES = ["PayFlow", "ShopEase", "LearnFlow", "TravelMate", "TeamDesk"]


def ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def append_jsonl(path: Path, record: dict) -> None:
    ensure_data_dir()
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def load_seed_feedback() -> list[dict]:
    return load_jsonl(SEED_FILE)


def load_live_feedback() -> list[dict]:
    return load_jsonl(LIVE_FILE)


def load_all_feedback() -> list[dict]:
    return load_seed_feedback() + load_live_feedback()


def load_product_changes() -> list[dict]:
    return load_jsonl(CHANGES_FILE)


def append_product_change(record: dict) -> None:
    append_jsonl(CHANGES_FILE, record)


def save_live_feedback(record: dict) -> None:
    append_jsonl(LIVE_FILE, record)


def clear_live_feedback() -> None:
    """GLOBAL wipe of every workspace's live feedback. Not used by the UI; the app
    must call clear_live_feedback_for_workspace() instead."""
    if LIVE_FILE.exists():
        LIVE_FILE.unlink()


def clear_live_feedback_for_workspace(workspace_id: str) -> int:
    """Remove live/imported feedback of ONE workspace only. Returns rows removed.

    Every other line (other workspaces, blank or unparseable lines) is written back
    byte-for-byte. Seed data, Hindsight, teach rules and product changes are never
    touched. An empty/blank id removes nothing.
    """
    wid = (workspace_id or "").strip() if isinstance(workspace_id, str) else ""
    if not wid or not LIVE_FILE.exists():
        return 0
    kept: list[str] = []
    removed = 0
    with LIVE_FILE.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if line:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    rec = None
                if isinstance(rec, dict) and rec.get("workspace") == wid:
                    removed += 1
                    continue
            kept.append(raw if raw.endswith("\n") else raw + "\n")
    if removed == 0:
        return 0
    tmp = LIVE_FILE.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.writelines(kept)
    tmp.replace(LIVE_FILE)
    return removed


# ---------------------------------------------------------------- workspaces
# Registry of custom workspaces: {workspace_id, display_name, created_at}.
# workspace_id is generated once and never changes; it is the ONLY key used for
# feedback, memories, Hindsight tags, teaching rules, product changes and UI state.
def _slug(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_-]+", "", re.sub(r"\s+", "", name or ""))
    return s[:40] or "ws"


def load_custom_workspaces() -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for r in load_jsonl(WORKSPACES_FILE):
        wid = str(r.get("workspace_id") or "").strip()
        if wid and wid not in seen and str(r.get("display_name") or "").strip():
            seen.add(wid)
            out.append(r)
    return out


def create_custom_workspace(display_name: str) -> dict:
    """Create a new custom workspace with a fresh unique immutable ID."""
    display_name = re.sub(r"\s+", " ", (display_name or "")).strip()
    existing = {r["workspace_id"] for r in load_custom_workspaces()} | set(BUILTIN_WORKSPACES)
    wid = f"{_slug(display_name)}__{uuid.uuid4().hex[:8]}"
    while wid in existing:
        wid = f"{_slug(display_name)}__{uuid.uuid4().hex[:8]}"
    rec = {
        "workspace_id": wid,
        "display_name": display_name,
        "created_at": datetime.utcnow().isoformat(timespec="seconds"),
    }
    append_jsonl(WORKSPACES_FILE, rec)
    return rec


def workspace_labels() -> dict[str, str]:
    """{workspace_id: label} in selector order. Duplicate names get ' (2)', ' (3)'..."""
    labels = {n: n for n in BUILTIN_WORKSPACES}
    used = {n.casefold(): 1 for n in BUILTIN_WORKSPACES}
    for r in load_custom_workspaces():
        name = r["display_name"].strip()
        k = name.casefold()
        used[k] = used.get(k, 0) + 1
        labels[r["workspace_id"]] = name if used[k] == 1 else f"{name} ({used[k]})"
    return labels


def workspace_display_name(workspace_id: str) -> str:
    """Friendly name for an ID (unnumbered). Unknown/legacy IDs are returned as-is."""
    for r in load_custom_workspaces():
        if r["workspace_id"] == workspace_id:
            return r["display_name"].strip()
    return workspace_id


def _norm_key(key: str) -> str:
    return key.strip().lower().replace(" ", "_").replace("-", "_")


FEEDBACK_KEYS = [
    "feedback_text", "feedbacktext", "text", "feedback", "review_text",
    "reviewtext", "review", "comment", "message", "content", "body",
    "description", "complaint", "note",
]
CHANNEL_KEYS = ["channel", "source", "type", "origin", "medium", "platform"]
DATE_KEYS = ["date", "timestamp", "created_at", "createdat", "time", "datetime"]
ID_KEYS = ["feedback_id", "feedbackid", "id", "review_id", "reviewid", "ticket_id", "ticketid"]
PERSONA_KEYS = ["persona", "user", "user_id", "userid", "customer", "customer_id", "customerid", "author"]


def _pick(row: dict, candidates: Iterable[str]) -> str | None:
    norm = {_norm_key(k): v for k, v in row.items()}
    for c in candidates:
        if c in norm and norm[c] not in (None, ""):
            return str(norm[c])
    return None


def detect_feedback_column(columns: list[str]) -> str | None:
    norm_map = {_norm_key(c): c for c in columns}
    for c in FEEDBACK_KEYS:
        if c in norm_map:
            return norm_map[c]
    return None


def normalize_imported_record(row: dict, workspace: str, source_type: str = "uploaded") -> dict | None:
    text = _pick(row, FEEDBACK_KEYS)
    if not text or not text.strip():
        return None
    return {
        "feedback_id": _pick(row, ID_KEYS) or str(uuid.uuid4()),
        "workspace": workspace,
        "channel": _pick(row, CHANNEL_KEYS) or "Imported",
        "timestamp": _pick(row, DATE_KEYS) or "",
        "text": text.strip(),
        "source_type": source_type,
        "persona": _pick(row, PERSONA_KEYS),
        "analysis": None,
        "lifecycle": "NOVEL",
    }