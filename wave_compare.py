"""Wave Comparison: every number here is computed with pandas / Python.

No LLM is involved. Taught rules (teach.py) are applied at read time, so the
comparison reflects corrected themes even for feedback stored before the
correction was taught.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from teach import find_matching_rule

UNKNOWN = "Unknown"
MODE_MONTH = "month"
MODE_WAVE_ID = "wave_id"


def _parse_dates(series: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(series, errors="coerce", utc=True, format="mixed")
    except (TypeError, ValueError):
        return pd.to_datetime(series, errors="coerce", utc=True)


def _key(theme: str) -> str:
    return str(theme).strip().lower()


def build_wave_frame(records: list[dict], rules: Optional[list[dict]] = None,
                     mode: str = MODE_MONTH) -> pd.DataFrame:
    """One row per record: wave (None if undated), theme, negative, reclassified."""
    rows = []
    for r in records or []:
        a = r.get("analysis") or {}
        stored = str(a.get("theme") or "").strip() or UNKNOWN
        theme, reclassified = stored, False
        if rules:
            hit = find_matching_rule(r.get("text") or "", rules)
            if hit:
                theme = hit["theme"]
                reclassified = _key(theme) != _key(stored)
        rows.append({
            "timestamp": str(r.get("timestamp") or ""),
            "wave_id": str(r.get("wave_id") or "").strip(),
            "theme": theme,
            "negative": str(a.get("sentiment") or "").strip().lower() == "negative",
            "reclassified": reclassified,
        })
    df = pd.DataFrame(rows, columns=["timestamp", "wave_id", "theme", "negative", "reclassified"])
    if df.empty:
        df["wave"] = pd.Series(dtype=object)
        return df

    if mode == MODE_WAVE_ID:
        wave = df["wave_id"].where(~df["wave_id"].str.lower().isin(["", "undated", "none", "nan"]))
    else:
        ts = _parse_dates(df["timestamp"])
        wave = ts.dt.strftime("%Y-%m")
    df["wave"] = wave.astype(object).where(wave.notna(), None)
    return df


def list_waves(df: pd.DataFrame) -> list[str]:
    if df.empty or "wave" not in df:
        return []
    return sorted(str(w) for w in df["wave"].dropna().unique())


def _theme_counts(sub: pd.DataFrame) -> tuple[dict, dict]:
    """(counts by theme-key, display label by theme-key); 'Unknown' excluded."""
    counts: dict[str, int] = {}
    labels: dict[str, str] = {}
    label_votes: dict[str, dict[str, int]] = {}
    for theme in sub["theme"]:
        k = _key(theme)
        if k == _key(UNKNOWN):
            continue
        counts[k] = counts.get(k, 0) + 1
        label_votes.setdefault(k, {})
        label_votes[k][theme] = label_votes[k].get(theme, 0) + 1
    for k, votes in label_votes.items():
        labels[k] = sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    return counts, labels


def _wave_stats(sub: pd.DataFrame) -> dict:
    total = int(len(sub))
    neg = int(sub["negative"].sum())
    return {
        "total": total,
        "negative": neg,
        "negative_pct": (neg / total * 100.0) if total else None,
        "no_theme": int((sub["theme"].map(_key) == _key(UNKNOWN)).sum()),
        "reclassified": int(sub["reclassified"].sum()),
    }


def compare_waves(df: pd.DataFrame, wave_a: Optional[str], wave_b: Optional[str],
                  top_n: int = 5) -> dict:
    """Compare two waves. Returns {"error": str} when the input can't be compared."""
    if df is None or df.empty:
        return {"error": "No feedback to compare."}
    if not wave_a or not wave_b:
        return {"error": "Select both Wave A and Wave B."}
    if wave_a == wave_b:
        return {"error": "Wave A and Wave B are the same wave. Pick two different waves."}
    a_df, b_df = df[df["wave"] == wave_a], df[df["wave"] == wave_b]
    if a_df.empty or b_df.empty:
        empty = wave_a if a_df.empty else wave_b
        return {"error": f"Wave {empty} has no feedback."}

    ca, la = _theme_counts(a_df)
    cb, lb = _theme_counts(b_df)
    labels = {**la, **lb}
    ta, tb = int(len(a_df)), int(len(b_df))

    rows = []
    for k in sorted(set(ca) | set(cb)):
        na, nb = ca.get(k, 0), cb.get(k, 0)
        if na == 0:
            status = "New"
        elif nb == 0:
            status = "Disappeared"
        elif nb > na:
            status = "Increased"
        elif nb < na:
            status = "Decreased"
        else:
            status = "Unchanged"
        rows.append({
            "Theme": labels[k],
            "Wave A count": na,
            "Wave B count": nb,
            "Change": nb - na,
            "Wave A share %": round(na / ta * 100, 1),
            "Wave B share %": round(nb / tb * 100, 1),
            "Status": status,
        })
    table = pd.DataFrame(rows, columns=["Theme", "Wave A count", "Wave B count", "Change",
                                        "Wave A share %", "Wave B share %", "Status"])
    if not table.empty:
        table = table.sort_values(["Change", "Theme"], ascending=[False, True],
                                  kind="stable").reset_index(drop=True)

    def top(counts: dict, lab: dict) -> pd.DataFrame:
        ordered = sorted(counts.items(), key=lambda kv: (-kv[1], lab[kv[0]]))[:top_n]
        return pd.DataFrame([{"Theme": lab[k], "Count": n} for k, n in ordered],
                            columns=["Theme", "Count"])

    def names(status: str) -> list[str]:
        return table.loc[table["Status"] == status, "Theme"].tolist() if not table.empty else []

    return {
        "error": None,
        "wave_a": wave_a,
        "wave_b": wave_b,
        "stats_a": _wave_stats(a_df),
        "stats_b": _wave_stats(b_df),
        "top_a": top(ca, la),
        "top_b": top(cb, lb),
        "table": table,
        "increased": table[table["Status"] == "Increased"] if not table.empty else table,
        "decreased": table[table["Status"] == "Decreased"] if not table.empty else table,
        "new": names("New"),
        "disappeared": names("Disappeared"),
        "unchanged": names("Unchanged"),
    }