"""Schema-agnostic feedback ingestion: load -> inspect -> detect -> map -> normalize.

All counts / statistics here are computed with Python / pandas only.
Normalized record shape:
    id, wave_id, timestamp, channel, segment, text, rating, metadata
"""
from __future__ import annotations

import io
import json
import re
import warnings
from datetime import datetime
from typing import Any, Optional

import pandas as pd


class IngestError(Exception):
    """A user-facing, recoverable ingestion problem."""


_ENCODINGS = ("utf-8-sig", "cp1252", "latin-1")
_NULLS = {"", "nan", "none", "null", "n/a", "na", "nil", "-", "--", "#n/a"}
_NUM_RE = r"(-?\d+(?:\.\d+)?)"

ROLE_LABELS = {
    "date_col": "Date / time",
    "rating_col": "Rating",
    "id_col": "ID",
    "channel_col": "Channel / source",
    "segment_col": "Segment / user / tier",
}

_DATE_W = {"date", "time", "timestamp", "datetime", "created", "posted", "submitted",
           "at", "day", "when", "published", "updated"}
_DATE_S = ("timestamp", "datetime", "createdat", "submittedat", "postedat")
_RATING_W = {"rating", "ratings", "score", "stars", "star", "nps", "csat", "grade", "rate", "rank"}
_RATING_S = ("rating", "stars", "csat")
_ID_W = {"id", "uuid", "guid", "ref", "reference", "key", "number", "no", "ticket", "index"}
_ID_S = ("uuid",)
_PERSON_W = {"user", "customer", "member", "client", "account", "persona"}
_SEGMENT_W = {"segment", "tier", "plan", "cohort", "persona", "user", "customer", "member",
              "role", "group", "client", "account", "subscription"}
_SEGMENT_STRONG = {"segment", "tier", "plan", "cohort", "persona"}
_SEGMENT_S = ("userid", "customerid", "segment")
_CHANNEL_W = {"channel", "source", "platform", "medium", "origin", "store", "touchpoint", "via"}
_CHANNEL_S = ("channel",)
_TEXT_W = {"feedback", "comment", "comments", "review", "reviews", "text", "message", "body",
           "description", "response", "verbatim", "note", "notes", "opinion", "complaint",
           "suggestion", "content", "remarks", "remark", "answer", "title", "subject",
           "summary", "details", "transcript", "reason", "story"}
_TEXT_S = ("feedback", "comment", "review", "verbatim")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _tokens(name: str) -> list[str]:
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(name))
    return [t for t in re.split(r"[^a-zA-Z0-9]+", s.lower()) if t]


def _hint(name: str, words=(), subs=()) -> bool:
    toks = _tokens(name)
    low = str(name).lower().replace("_", "").replace(" ", "")
    return any(t in words for t in toks) or any(sb in low for sb in subs)


def _null_mask(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin(_NULLS)


def _isnull(v: Any) -> bool:
    return str(v).strip().lower() in _NULLS


def _nonnull(s: pd.Series) -> pd.Series:
    return s[~_null_mask(s)].astype(str).str.strip()


def _cell(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and v != v:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False)
    return str(v)


def _parse_dates(series: pd.Series) -> pd.Series:
    """Parse to UTC datetimes; unparseable -> NaT. Plain small numbers are never dates."""
    s = series.astype(str).str.strip()
    num = pd.to_numeric(s, errors="coerce")
    if len(s) and num.notna().mean() > 0.8:
        med = num.median()
        unit = "ms" if med >= 1e11 else ("s" if med >= 1e8 else None)
        if unit is None:
            return pd.to_datetime(pd.Series([pd.NaT] * len(s), index=s.index), utc=True)
        return pd.to_datetime(num, unit=unit, errors="coerce", utc=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return pd.to_datetime(s, errors="coerce", utc=True, format="mixed")
        except (TypeError, ValueError):
            return pd.to_datetime(s, errors="coerce", utc=True)


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def _decode(data: bytes) -> tuple[str, str]:
    for enc in _ENCODINGS:
        try:
            return data.decode(enc), ("utf-8" if enc == "utf-8-sig" else enc)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1", errors="replace"), "latin-1"


def _read_csv_text(text: str) -> pd.DataFrame:
    lines = text.lstrip("\ufeff").splitlines()
    header = lines[0] if lines else ""
    counts = {d: header.count(d) for d in (",", ";", "\t", "|")}
    sep = max(counts, key=lambda d: (counts[d], d == ","))
    if counts[sep] == 0:
        sep = ","
    try:
        return pd.read_csv(
            io.StringIO(text), sep=sep, dtype=str, keep_default_na=False,
            skip_blank_lines=True, on_bad_lines="skip",
        )
    except pd.errors.EmptyDataError:
        raise IngestError("The file has no header or data rows.")
    except Exception as e:
        raise IngestError(f"This does not look like a valid CSV file ({e}).")


def _records_from_json(text: str, is_lines: bool) -> list[dict]:
    try:
        if is_lines:
            recs = [json.loads(l) for l in text.splitlines() if l.strip()]
        else:
            data = json.loads(text)
            if isinstance(data, dict):
                lists = [v for v in data.values()
                         if isinstance(v, list) and v and all(isinstance(x, dict) for x in v)]
                data = lists[0] if len(lists) == 1 else [data]
            recs = data
    except json.JSONDecodeError as e:
        raise IngestError(f"Could not parse the JSON file: {e}")
    if not isinstance(recs, list) or not recs or not all(isinstance(r, dict) for r in recs):
        raise IngestError("The JSON file must contain a list of objects (one per feedback item).")
    return recs


def _clean_frame(df: pd.DataFrame, info: dict) -> pd.DataFrame:
    if df.shape[0] == 0 or df.shape[1] == 0:
        raise IngestError("The file has a header but no data rows.")
    names, seen = [], {}
    for i, c in enumerate(df.columns):
        c = str(c).strip()
        if not c or c.lower().startswith("unnamed:"):
            c = f"column_{i + 1}"
        k = seen.get(c, 0)
        seen[c] = k + 1
        names.append(c if k == 0 else f"{c}_{k + 1}")
    df = df.copy()
    df.columns = names
    df = df.fillna("").astype(str).apply(lambda s: s.str.strip())
    null = df.apply(_null_mask)
    blank_rows = null.all(axis=1)
    info["blank_rows_dropped"] = int(blank_rows.sum())
    df = df.loc[~blank_rows]
    null = null.loc[~blank_rows]
    df = df[[c for c in df.columns if not null[c].all()]].reset_index(drop=True)
    if df.shape[0] == 0 or df.shape[1] == 0:
        raise IngestError("The file contains no non-empty rows.")
    info["rows"] = int(len(df))
    return df


def load_table(filename: str, data: bytes) -> tuple[pd.DataFrame, dict]:
    """Load CSV / JSON / JSONL bytes into an all-string DataFrame. Raises IngestError."""
    if not data or not data.strip():
        raise IngestError("The file is empty.")
    text, enc = _decode(data)
    info: dict = {"encoding": enc, "rows": 0, "blank_rows_dropped": 0}
    name = (filename or "").lower()
    if name.endswith(".jsonl"):
        df = pd.DataFrame(_records_from_json(text, True))
        df = df.apply(lambda s: s.map(_cell))
    elif name.endswith(".json"):
        df = pd.DataFrame(_records_from_json(text, False))
        df = df.apply(lambda s: s.map(_cell))
    else:
        df = _read_csv_text(text)
    return _clean_frame(df, info), info


# --------------------------------------------------------------------------
# inspection + detection
# --------------------------------------------------------------------------
def profile_columns(df: pd.DataFrame) -> dict[str, dict]:
    prof: dict[str, dict] = {}
    n = len(df)
    for c in df.columns:
        vals = _nonnull(df[c])
        ne = len(vals)
        if ne == 0:
            prof[c] = {"non_empty_ratio": 0.0, "unique_ratio": 0.0, "nunique": 0, "avg_len": 0.0,
                       "avg_words": 0.0, "numeric_ratio": 0.0, "num_min": None, "num_max": None,
                       "date_ratio": 0.0, "samples": []}
            continue
        num = pd.to_numeric(vals, errors="coerce")
        samples = [v if len(v) <= 60 else v[:57] + "..." for v in vals.drop_duplicates().head(3)]
        prof[c] = {
            "non_empty_ratio": ne / n if n else 0.0,
            "unique_ratio": vals.nunique() / ne,
            "nunique": int(vals.nunique()),
            "avg_len": float(vals.str.len().mean()),
            "avg_words": float(vals.str.split().str.len().mean()),
            "numeric_ratio": float(num.notna().mean()),
            "num_min": float(num.min()) if num.notna().any() else None,
            "num_max": float(num.max()) if num.notna().any() else None,
            "date_ratio": float(_parse_dates(vals.head(300)).notna().mean()),
            "samples": samples,
        }
    return prof


def detect_mapping(df: pd.DataFrame, profile: Optional[dict] = None) -> dict:
    """Guess roles. Returns text_cols (list) + date/rating/id/channel/segment (col or None)."""
    prof = profile or profile_columns(df)
    cols = list(df.columns)
    used: set[str] = set()

    def free() -> list[str]:
        return [c for c in cols if c not in used]

    # date
    date_col, best = None, 0.0
    for c in free():
        p = prof[c]
        hinted = _hint(c, _DATE_W, _DATE_S)
        if p["date_ratio"] < 0.8 or p["avg_len"] < 6 or p["avg_words"] > 6:
            continue
        if p["numeric_ratio"] > 0.8 and not hinted:
            continue
        score = p["date_ratio"] + (0.5 if hinted else 0.0)
        if score > best:
            best, date_col = score, c
    if date_col:
        used.add(date_col)

    # rating
    rating_col, best = None, 0
    for c in free():
        p = prof[c]
        if p["avg_words"] > 3:
            continue
        nums = pd.to_numeric(_nonnull(df[c]).head(500).str.extract(_NUM_RE)[0], errors="coerce")
        if not len(nums) or nums.notna().mean() < 0.8:
            continue
        if _hint(c, _RATING_W, _RATING_S):
            score = 2
        elif (p["numeric_ratio"] >= 0.95 and 2 <= p["nunique"] <= 11
              and p["num_min"] is not None and p["num_min"] >= 0 and p["num_max"] <= 10
              and p["unique_ratio"] <= 0.3 and not _hint(c, _ID_W)):
            score = 1
        else:
            continue
        if score > best:
            best, rating_col = score, c
    if rating_col:
        used.add(rating_col)

    # id
    id_col, best = None, 0
    for c in free():
        p = prof[c]
        if not _hint(c, _ID_W, _ID_S) or _hint(c, _PERSON_W):
            continue
        if p["unique_ratio"] < 0.9 or p["non_empty_ratio"] < 0.9 or p["avg_words"] > 3:
            continue
        score = 2 if c.strip().lower() in {"id", "uuid"} else 1
        if score > best:
            best, id_col = score, c
    if id_col:
        used.add(id_col)

    # segment / user / tier
    segment_col, best = None, 0
    for c in free():
        p = prof[c]
        if not _hint(c, _SEGMENT_W, _SEGMENT_S) or p["avg_words"] > 4:
            continue
        score = 2 if any(t in _SEGMENT_STRONG for t in _tokens(c)) else 1
        if score > best:
            best, segment_col = score, c
    if segment_col:
        used.add(segment_col)

    # channel / source
    channel_col = None
    for c in free():
        p = prof[c]
        if _hint(c, _CHANNEL_W, _CHANNEL_S) and p["avg_words"] <= 4:
            channel_col = c
            break
    if channel_col:
        used.add(channel_col)

    # text (one or more)
    text_cols: list[str] = []
    for c in free():
        p = prof[c]
        if p["numeric_ratio"] >= 0.5 or p["non_empty_ratio"] == 0:
            continue
        hinted = _hint(c, _TEXT_W, _TEXT_S) and p["avg_words"] >= 1.5 and (
            p["unique_ratio"] >= 0.3 or p["avg_words"] >= 5)
        free_text = p["avg_words"] >= 4 and p["avg_len"] >= 20 and p["unique_ratio"] >= 0.3
        if hinted or free_text:
            text_cols.append(c)
    if not text_cols:
        cand = [c for c in free() if prof[c]["numeric_ratio"] < 0.5 and prof[c]["avg_words"] >= 2]
        if cand:
            text_cols = [max(cand, key=lambda c: prof[c]["avg_words"])]

    return {
        "text_cols": text_cols,
        "date_col": date_col,
        "rating_col": rating_col,
        "id_col": id_col,
        "channel_col": channel_col,
        "segment_col": segment_col,
    }


def validate_mapping(df: pd.DataFrame, mapping: dict) -> list[str]:
    problems: list[str] = []
    text_cols = mapping.get("text_cols") or []
    if not text_cols:
        problems.append(
            "No feedback text column is selected. Choose at least one column that contains "
            "the actual customer comments to continue."
        )
    seen: dict[str, str] = {}
    for role, label in ROLE_LABELS.items():
        c = mapping.get(role)
        if not c:
            continue
        if c in text_cols:
            problems.append(f"'{c}' is used as both feedback text and {label}. Use each column once.")
        elif c in seen:
            problems.append(f"'{c}' is mapped to both {seen[c]} and {label}. Use each column once.")
        else:
            seen[c] = label
    return problems


# --------------------------------------------------------------------------
# normalization
# --------------------------------------------------------------------------
def normalize_table(df: pd.DataFrame, mapping: dict,
                    upload_time: Optional[datetime] = None) -> tuple[list[dict], dict]:
    """Return (records, stats). Rows without usable text are skipped and counted."""
    upload_iso = (upload_time or datetime.utcnow()).isoformat(timespec="seconds")
    n = len(df)
    text_cols = list(mapping.get("text_cols") or [])

    # text (multiple columns joined as "Column: value")
    parts = []
    for c in text_cols:
        s = df[c].astype(str).str.strip()
        parts.append((c, s.where(~_null_mask(s), "").tolist()))
    if len(parts) == 1:
        texts = parts[0][1]
    else:
        texts = ["\n".join(f"{c}: {v[i]}" for c, v in parts if v[i]) for i in range(n)]

    # timestamp / wave
    date_col = mapping.get("date_col")
    dt = _parse_dates(df[date_col].where(~_null_mask(df[date_col]), "")) if date_col else None
    valid = dt.notna() if dt is not None else None
    dated = bool(valid is not None and valid.any())
    if not dated:
        waves = ["wave-1"] * n
        stamps = [upload_iso] * n
        date_min = date_max = None
    else:
        vdt = dt[valid]
        span_days = (vdt.max() - vdt.min()).days
        date_min = vdt.min().strftime("%Y-%m-%d")
        date_max = vdt.max().strftime("%Y-%m-%d")
        if span_days >= 60:
            lab = dt.dt.strftime("%Y-%m")
        elif span_days >= 14:
            iso = dt.dt.isocalendar()
            lab = iso["year"].astype("string") + "-W" + iso["week"].astype("string").str.zfill(2)
        else:
            lab = pd.Series(["wave-1"] * n, index=dt.index)
        strf = dt.dt.strftime("%Y-%m-%dT%H:%M:%S")
        ok = valid.tolist()
        waves = [str(l) if ok[i] and isinstance(l, str) else "undated"
                 for i, l in enumerate(lab.tolist())]
        stamps = [s if ok[i] and isinstance(s, str) else upload_iso
                  for i, s in enumerate(strf.tolist())]

    # rating
    rating_col = mapping.get("rating_col")
    if rating_col:
        ratings = pd.to_numeric(
            df[rating_col].astype(str).str.extract(_NUM_RE)[0], errors="coerce").tolist()
    else:
        ratings = [None] * n

    def col_list(name: Optional[str]) -> list[str]:
        if not name:
            return [""] * n
        s = df[name].astype(str).str.strip()
        return s.where(~_null_mask(s), "").tolist()

    ids_raw = col_list(mapping.get("id_col"))
    channels = col_list(mapping.get("channel_col"))
    segments = col_list(mapping.get("segment_col"))
    raw_dates = col_list(date_col)

    mapped = set(text_cols) | {mapping.get(r) for r in ROLE_LABELS if mapping.get(r)}
    meta_cols = [c for c in df.columns if c not in mapped]
    meta_lists = {c: df[c].astype(str).str.strip().tolist() for c in meta_cols}

    records: list[dict] = []
    seen_ids: dict[str, int] = {}
    for i in range(n):
        text = texts[i].strip()
        if not text:
            continue
        rid = ids_raw[i] or f"row-{i + 1}"
        k = seen_ids.get(rid, 0)
        seen_ids[rid] = k + 1
        if k:
            rid = f"{rid}-{k + 1}"
        meta = {c: v[i] for c, v in meta_lists.items() if not _isnull(v[i])}
        if dated and waves[i] == "undated" and raw_dates[i]:
            meta["raw_timestamp"] = raw_dates[i]
        rating = ratings[i]
        records.append({
            "id": rid,
            "wave_id": waves[i],
            "timestamp": stamps[i],
            "channel": channels[i] or "Imported",
            "segment": segments[i],
            "text": text,
            "rating": None if rating is None or rating != rating else float(rating),
            "metadata": meta,
        })

    stats = {
        "rows_total": n,
        "rows_kept": len(records),
        "rows_without_text": n - len(records),
        "dated": dated,
        "date_min": date_min,
        "date_max": date_max,
    }
    return records, stats


def summarize_records(records: list[dict]) -> dict:
    """All numbers come from pandas over the normalized records."""
    if not records:
        return {"total": 0, "waves": {}, "channels": {}, "segments": {},
                "rating_count": 0, "rating_mean": None, "rating_distribution": {}}
    df = pd.DataFrame(records)
    rating = pd.to_numeric(df["rating"], errors="coerce")
    seg = df["segment"].astype(str).str.strip()
    return {
        "total": int(len(df)),
        "waves": {str(k): int(v) for k, v in df["wave_id"].value_counts().sort_index().items()},
        "channels": {str(k): int(v) for k, v in df["channel"].value_counts().items()},
        "segments": {str(k): int(v) for k, v in seg[seg != ""].value_counts().items()},
        "rating_count": int(rating.notna().sum()),
        "rating_mean": float(rating.mean()) if rating.notna().any() else None,
        "rating_distribution": {str(k): int(v) for k, v in rating.dropna().value_counts().sort_index().items()},
    }


# --------------------------------------------------------------------------
# display helpers
# --------------------------------------------------------------------------
def profile_table(profile: dict) -> pd.DataFrame:
    rows = []
    for c, p in profile.items():
        rows.append({
            "Column": c,
            "Sample values": " | ".join(p["samples"]),
            "Non-empty": f"{p['non_empty_ratio'] * 100:.0f}%",
            "Avg words": round(p["avg_words"], 1),
        })
    return pd.DataFrame(rows)


def mapping_preview(mapping: dict, df: pd.DataFrame) -> pd.DataFrame:
    def example(c: str) -> str:
        v = _nonnull(df[c])
        if not len(v):
            return ""
        s = v.iloc[0]
        return s if len(s) <= 70 else s[:67] + "..."

    rows = [{"Role": "Feedback text", "Column": c, "Example": example(c)}
            for c in (mapping.get("text_cols") or [])]
    for role, label in ROLE_LABELS.items():
        c = mapping.get(role)
        rows.append({"Role": label, "Column": c or "— not present —", "Example": example(c) if c else ""})
    return pd.DataFrame(rows)


def preview_frame(records: list[dict], n: int = 5) -> pd.DataFrame:
    rows = []
    for r in records[:n]:
        rows.append({
            "id": r["id"],
            "wave_id": r["wave_id"],
            "timestamp": r["timestamp"],
            "channel": r["channel"],
            "segment": r["segment"],
            "text": r["text"] if len(r["text"]) <= 120 else r["text"][:117] + "...",
            "rating": r["rating"],
            "metadata": json.dumps(r["metadata"], ensure_ascii=False)[:120],
        })
    return pd.DataFrame(rows)