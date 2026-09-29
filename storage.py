import json
import uuid
from pathlib import Path
from typing import Iterable

DATA_DIR = Path("data")
SEED_FILE = DATA_DIR / "seed_feedback.jsonl"
LIVE_FILE = DATA_DIR / "live_feedback.jsonl"
CHANGES_FILE = DATA_DIR / "product_changes.jsonl"


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
    if LIVE_FILE.exists():
        LIVE_FILE.unlink()


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
