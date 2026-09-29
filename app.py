import hashlib
import json
import pandas as pd
import streamlit as st

from config import groq_ready, hindsight_ready
from storage import (
    DATA_DIR,
    BUILTIN_WORKSPACES,
    load_seed_feedback,
    load_live_feedback,
    load_all_feedback,
    load_product_changes,
    clear_live_feedback_for_workspace,
    create_custom_workspace,
    workspace_labels,
    detect_feedback_column,
)
from seed_data import generate_seed_if_missing
from pipeline import process_feedback, import_records, import_normalized, record_product_change
from csv_ingest import (
    IngestError,
    load_table,
    profile_columns,
    detect_mapping,
    validate_mapping,
    normalize_table,
    summarize_records,
    profile_table,
    mapping_preview,
    preview_frame,
)
from voice import transcribe_audio, VoiceUnavailable
from llm import GroqUnavailable
from wave_teach_ui import render_wave_comparison, render_teach_agent

st.set_page_config(page_title="Feedback Memory Intelligence Agent", layout="wide")

# --- Bootstrap seed data
if not (DATA_DIR / "seed_feedback.jsonl").exists():
    generate_seed_if_missing()

# --- Workspaces
DEMO_WORKSPACES = list(BUILTIN_WORKSPACES)  # single source of truth lives in storage.py

# Workspace IDs are the only internal key. Built-ins keep ID == name; custom
# workspaces get a unique immutable ID stored in data/workspaces.jsonl, so the
# mapping survives reruns and restarts. Labels are for display only.
_WS_LABELS = workspace_labels()  # {workspace_id: friendly label}
all_workspaces = list(_WS_LABELS)


def ws_label(ws: str) -> str:
    """Friendly display name for a workspace_id (unknown/legacy IDs shown as-is)."""
    return _WS_LABELS.get(ws, ws)


def create_workspace():
    clean_name = st.session_state["new_workspace_name"].strip()

    if not clean_name:
        st.session_state["workspace_message"] = "Enter a workspace name."
        return

    if clean_name.casefold() in {n.casefold() for n in DEMO_WORKSPACES}:
        st.session_state["workspace_message"] = (
            f"Workspace '{clean_name}' already exists."
        )
        return

    rec = create_custom_workspace(clean_name)

    # This is allowed because it happens inside the button callback.
    st.session_state["active_workspace"] = rec["workspace_id"]

    st.session_state["workspace_message"] = (
        f"Created workspace: {workspace_labels().get(rec['workspace_id'], clean_name)}"
    )


# --- Sidebar
st.sidebar.title("🧠 Feedback Memory Agent")

st.sidebar.markdown("### Workspace")

# If the remembered selection is no longer a known workspace_id, fall back to the
# first one instead of crashing (never guess/remap by display name).
if st.session_state.get("active_workspace") not in all_workspaces:
    st.session_state["active_workspace"] = all_workspaces[0]

workspace = st.sidebar.selectbox(
    "Select workspace",
    all_workspaces,
    key="active_workspace",
    format_func=ws_label,
)

with st.sidebar.expander("➕ Create New Workspace"):
    st.text_input(
        "Workspace name",
        placeholder="e.g. PhonePe, Amazon, Netflix",
        key="new_workspace_name",
    )

    st.button(
        "Create Workspace",
        key="create_workspace_btn",
        on_click=create_workspace,
    )

    if st.session_state.get("workspace_message"):
        st.sidebar.caption(
            st.session_state["workspace_message"]
        )

# --- Sidebar
st.sidebar.title("🧠 Feedback Memory Agent")


groq_status = "✅ Groq configured" if groq_ready() else "⚠ Groq key missing"
hindsight_status = "✅ Hindsight configured" if hindsight_ready() else "⚠ Hindsight keys missing"
st.sidebar.caption(groq_status)
st.sidebar.caption(hindsight_status)

page = st.sidebar.radio(
    "Navigate",
    [
        "📥 Feedback Inbox",
        "🧠 Hindsight Memory",
        "🕰️ Feedback Time Machine",
        "📊 Product Intelligence",
        "🎤 Live Feedback",
        "🧪 Judge Test Mode",
    ],
)


def ws_feedback(ws: str) -> list[dict]:
    return [f for f in load_all_feedback() if f.get("workspace") == ws]


def memory_is_on(ws: str) -> bool:
    """Agent Memory toggle state for a workspace (default ON, per browser session)."""
    return st.session_state.setdefault("memory_enabled_by_ws", {}).get(ws, True)


def _set_memory_toggle(ws: str) -> None:
    st.session_state.setdefault("memory_enabled_by_ws", {})[ws] = bool(
        st.session_state.get(f"memory_toggle_{ws}", True)
    )


def last_result_for(ws: str):
    """The most recent agent result, but ONLY if it belongs to workspace `ws`."""
    last = st.session_state.get("last_result")
    if not last:
        return None
    owner = last.get("workspace") or getattr(last.get("item"), "workspace", None)
    return last if owner == ws else None


def count_line(ws: str) -> str:
    seed = [f for f in load_seed_feedback() if f.get("workspace") == ws]
    live = [f for f in load_live_feedback() if f.get("workspace") == ws]
    return f"Seed: {len(seed)} | Live/Imported: {len(live)} | Total: {len(seed) + len(live)}"


# =========================================================================
# Product Intelligence Agent — grounded Q&A helpers
# Every count/trend is computed with pandas from the selected workspace's
# stored feedback. No LLM is used for numbers or wording, so nothing can be
# invented. Hindsight memories are only ever shown as recalled, never rewritten.
# =========================================================================
import inspect
import re

PI_SUGGESTED_QUESTIONS = [
    "What are the top feedback themes?",
    "Why are customers unhappy?",
    "Which channel has the most complaints?",
    "What recurring issues are we seeing?",
    "What did the agent learn from previous feedback?",
]

_PI_COLS = ["Date", "Channel", "Feedback", "Theme", "Sentiment", "Priority", "Lifecycle"]
_PI_STOPWORDS = {
    "show", "evidence", "for", "the", "issue", "issues", "problem", "problems", "of", "on",
    "about", "me", "us", "any", "example", "examples", "feedback", "related", "with",
    "customer", "customers", "complaint", "complaints", "what", "are", "there", "give",
    "see", "find", "regarding", "and", "proof", "quotes", "records", "from", "our",
}


def _pi_parse_dates(series: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(series, errors="coerce", utc=True, format="mixed")
    except (TypeError, ValueError):
        return pd.to_datetime(series, errors="coerce", utc=True)


def _pi_frame(records: list[dict]) -> pd.DataFrame:
    rows = []
    for r in records:
        a = r.get("analysis") or {}
        rows.append({
            "Date": r.get("timestamp") or "",
            "Channel": r.get("channel") or "Unknown",
            "Feedback": r.get("text") or "",
            "Theme": a.get("theme") or "Unknown",
            "Sentiment": a.get("sentiment") or "Unknown",
            "Priority": a.get("priority") or "Unknown",
            "Lifecycle": r.get("lifecycle") or "NOVEL",
        })
    df = pd.DataFrame(rows, columns=_PI_COLS)
    for c in ("Channel", "Theme", "Sentiment", "Priority", "Lifecycle"):
        df[c] = df[c].astype(str)
    df["_ts"] = _pi_parse_dates(df["Date"].astype(str))
    return df


def _pi_pct(n: int, d: int) -> str:
    return f"{n / d * 100:.1f}%" if d else "n/a"


def _pi_sample(df: pd.DataFrame, limit: int = 8) -> pd.DataFrame:
    if df.empty:
        return df
    return df.sort_values("_ts", ascending=False, na_position="last", kind="stable").head(limit)


def _pi_per_group(df: pd.DataFrame, col: str, groups, per: int = 2) -> pd.DataFrame:
    parts = [_pi_sample(df[df[col] == g], per) for g in groups]
    return pd.concat(parts) if parts else df.iloc[0:0]


def _pi_result(answer, data=None, relevant=None, memories=None, memory_note=""):
    return {
        "answer": answer,
        "data": data,
        "relevant": relevant,
        "memories": memories or [],
        "memory_note": memory_note,
    }


# --- Hindsight recall (workspace-scoped) ---------------------------------
def _pi_field(obj, *names, default=""):
    for n in names:
        v = obj.get(n) if isinstance(obj, dict) else getattr(obj, n, None)
        if v not in (None, ""):
            return v
    return default


def recall_workspace_memories(ws: str, query: str, limit: int = 5):
    """Return (memories, note). Calls HindsightClient.recall() directly, scoped to this
    workspace only. Returns [] plus an honest note if Memory is OFF or Hindsight is unavailable."""
    if not memory_is_on(ws):
        return [], "Agent Memory is OFF for this workspace, so no Hindsight memories were recalled."
    if not hindsight_ready():
        return [], "Hindsight is not configured, so no memories were recalled."
    try:
        from hindsight_client import HindsightClient, HindsightUnavailable
    except Exception as e:  # noqa: BLE001
        return [], f"Hindsight client could not be loaded ({e})."

    try:
        raw = HindsightClient().recall(
            query=str(query or "customer feedback"),
            workspace=ws,
            top_k=limit,
        )
    except HindsightUnavailable as e:
        return [], f"Hindsight is unavailable ({e})."
    except Exception as e:  # noqa: BLE001
        return [], f"Hindsight recall failed ({e})."

    memories = []
    for item in list(raw or [])[:limit]:
        m = getattr(item, "memory", item)
        m_ws = _pi_field(m, "workspace", default=None)
        if not m_ws or str(m_ws).strip() != ws.strip():
            continue  # never surface another workspace's (or an unscoped) memory
        text = _pi_field(m, "raw_text", "text")
        if not text:
            continue
        memories.append({
            "text": str(text),
            "timestamp": str(_pi_field(m, "timestamp")),
            "channel": str(_pi_field(m, "channel")),
            "theme": str(_pi_field(m, "discovered_theme", "theme")),
            "issue": str(_pi_field(m, "underlying_issue")),
        })
    if not memories:
        return [], "Hindsight returned no memories for this workspace and question."
    return memories, f"{len(memories)} memory(ies) recalled from Hindsight for this workspace."


# --- Question handlers (all numbers via pandas) --------------------------
def _pi_themes(df, ws):
    total = len(df)
    vc = df["Theme"].value_counts()
    top = vc.head(5)
    lines = [
        f"{i}. **{t}** — {int(n)} of {total} records ({_pi_pct(int(n), total)})"
        for i, (t, n) in enumerate(top.items(), start=1)
    ]
    ans = f"Top feedback themes across {total} stored records in **{ws_label(ws)}**:\n\n" + "\n".join(lines)
    if len(vc) > 5:
        ans += f"\n\n{len(vc) - 5} other theme(s) not shown."
    data = _pi_per_group(df, "Theme", list(top.index), 2)
    return _pi_result(ans, data, relevant=int(top.sum()))


def _pi_unhappy(df, ws):
    total = len(df)
    neg = df[df["Sentiment"].str.lower() == "negative"]
    if neg.empty:
        return _pi_result(f"None of the {total} stored records in **{ws_label(ws)}** are labelled negative, so there is no stored evidence of unhappiness.")
    vc = neg["Theme"].value_counts().head(3)
    high = int(neg["Priority"].str.lower().isin(["high", "critical"]).sum())
    lines = [f"- **{t}** — {int(n)} of {len(neg)} negative records ({_pi_pct(int(n), len(neg))})" for t, n in vc.items()]
    ans = (
        f"**{len(neg)} of {total}** stored records ({_pi_pct(len(neg), total)}) are labelled negative; "
        f"{high} of them {'is' if high == 1 else 'are'} high/critical priority. Themes driving them:\n\n" + "\n".join(lines)
    )
    data = _pi_per_group(neg, "Theme", list(vc.index), 3)
    return _pi_result(ans, data, relevant=len(neg))


def _pi_channel(df, ws):
    g = (
        df.assign(_neg=df["Sentiment"].str.lower() == "negative")
        .groupby("Channel")
        .agg(neg=("_neg", "sum"), total=("_neg", "size"))
        .sort_values(["neg", "total"], ascending=False)
    )
    if int(g["neg"].sum()) == 0:
        return _pi_result(f"No stored record in **{ws_label(ws)}** is labelled negative, so no channel has complaints.")
    top_n = int(g["neg"].iloc[0])
    leaders = list(g.index[g["neg"] == top_n])
    lines = [
        f"- **{c}** — {int(r.neg)} negative of {int(r.total)} ({_pi_pct(int(r.neg), int(r.total))})"
        for c, r in g.head(5).iterrows()
    ]
    lead = ", ".join(f"**{c}**" for c in leaders)
    tie = " (tied)" if len(leaders) > 1 else ""
    verb = "have" if len(leaders) > 1 else "has"
    ans = (
        f"{lead}{tie} {verb} the most complaints with {top_n} negative record(s). "
        "Here \"complaints\" means records labelled negative.\n\n" + "\n".join(lines)
    )
    neg_rows = df[(df["Channel"].isin(leaders)) & (df["Sentiment"].str.lower() == "negative")]
    return _pi_result(ans, _pi_sample(neg_rows, 8), relevant=len(neg_rows))


def _pi_recurring(df, ws):
    total = len(df)
    rec = df[df["Lifecycle"].str.upper() == "RECURRING"]
    mems, note = recall_workspace_memories(ws, "recurring customer issues")
    if rec.empty:
        return _pi_result(
            f"No stored record in **{ws_label(ws)}** is currently marked RECURRING (out of {total}).",
            memories=mems, memory_note=note,
        )
    vc = rec["Theme"].value_counts().head(5)
    lines = [f"- **{t}** — {int(n)} recurring record(s)" for t, n in vc.items()]
    ans = (
        f"**{len(rec)} of {total}** stored records ({_pi_pct(len(rec), total)}) are marked RECURRING. "
        "By theme:\n\n" + "\n".join(lines)
    )
    return _pi_result(ans, _pi_per_group(rec, "Theme", list(vc.index), 2), relevant=len(rec),
                      memories=mems, memory_note=note)


def _pi_learned(df, ws):
    total = len(df)
    top_theme = df["Theme"].value_counts()
    top_channel = df["Channel"].value_counts()
    rec = int((df["Lifecycle"].str.upper() == "RECURRING").sum())
    mems, note = recall_workspace_memories(ws, "customer feedback themes and recurring issues")
    ans = (
        f"From the {total} stored records in **{ws_label(ws)}**: the most frequent theme is "
        f"**{top_theme.index[0]}** ({int(top_theme.iloc[0])} records), the largest source is "
        f"**{top_channel.index[0]}** ({int(top_channel.iloc[0])} records), and {rec} record(s) are marked RECURRING."
    )
    if mems:
        ans += f"\n\nHindsight also recalled {len(mems)} historical experience(s) — see Evidence → Memory."
    else:
        ans += f"\n\nNo Hindsight memories are shown: {note}"
    top = _pi_sample(df[df["Theme"] == top_theme.index[0]], 5)
    return _pi_result(ans, top, relevant=int(top_theme.iloc[0]), memories=mems, memory_note=note)


def _pi_changes(df, ws):
    changes = [pc for pc in load_product_changes() if pc.get("workspace") == ws]
    if not changes:
        return _pi_result(f"No product changes are recorded for **{ws_label(ws)}**, so there is nothing to compare before/after.")
    for pc in changes:
        pc["_d"] = pd.to_datetime(pc.get("date"), errors="coerce", utc=True)
    changes = sorted(changes, key=lambda p: (pd.isna(p["_d"]), p["_d"] if not pd.isna(p["_d"]) else 0), reverse=True)[:5]

    lines, after_frames = [], []
    for pc in changes:
        head = (
            f"- **{pc.get('title')}** (v{pc.get('version')}, {pc.get('date')}) — addressed: {pc.get('problem_issue')}. "
            f"Expected: {pc.get('expected_outcome')}. Observed (as recorded): {pc.get('observed_outcome') or 'not recorded'}."
        )
        theme = str(pc.get("linked_theme") or "").strip()
        if not theme:
            lines.append(head + " No linked theme recorded, so no before/after counts.")
            continue
        th = df[df["Theme"].str.lower() == theme.lower()]
        if th.empty:
            lines.append(head + f" No stored feedback has the linked theme \"{theme}\".")
            continue
        if pd.isna(pc["_d"]):
            lines.append(head + " The change date could not be read, so no before/after counts.")
            continue
        dated = th[th["_ts"].notna()]
        before, after = dated[dated["_ts"] < pc["_d"]], dated[dated["_ts"] >= pc["_d"]]
        neg = lambda x: int((x["Sentiment"].str.lower() == "negative").sum())  # noqa: E731
        undated = len(th) - len(dated)
        extra = f", {undated} undated not counted" if undated else ""
        lines.append(
            head + f" Linked theme **{theme}**: before {len(before)} record(s) ({neg(before)} negative), "
            f"on/after {len(after)} ({neg(after)} negative){extra}."
        )
        after_frames.append(after)

    ans = (
        f"Product changes recorded for **{ws_label(ws)}** with stored feedback counts around each date:\n\n" + "\n".join(lines)
        + "\n\nCounts are raw and the before/after periods may differ in length; this shows timing, not proof of cause."
    )
    data = None
    if after_frames:
        data = pd.concat(after_frames)
        data = _pi_sample(data[~data.index.duplicated()], 8)
    themes = " ".join(str(pc.get("linked_theme") or "") for pc in changes).strip()
    mems, note = recall_workspace_memories(ws, f"product change {themes}".strip())
    return _pi_result(ans, data, relevant=None if data is None else len(data), memories=mems, memory_note=note)


def _pi_evidence(df, q, ws):
    ql = q.lower()
    kws = [w for w in re.findall(r"[a-z0-9]+", ql) if w not in _PI_STOPWORDS and len(w) > 2]
    label = None
    theme_hit = [t for t in df["Theme"].unique() if t.lower() != "unknown" and t.lower() in ql]
    if theme_hit:
        label = ", ".join(theme_hit)
        hits = df[df["Theme"].isin(theme_hit)]
    elif kws:
        label = " ".join(kws)
        hay = (df["Feedback"] + " " + df["Theme"]).str.lower()
        mask = pd.Series(True, index=df.index)
        for k in kws:
            mask &= hay.str.contains(k, regex=False)
        hits = df[mask]
        if hits.empty and len(kws) > 1:
            mask = pd.Series(False, index=df.index)
            for k in kws:
                mask |= hay.str.contains(k, regex=False)
            hits = df[mask]
    else:
        return _pi_result("Tell me which topic to look for, e.g. \"Show evidence for the refund issue\".")

    if hits.empty:
        return _pi_result(f"No stored feedback in **{ws_label(ws)}** matches \"{label}\", so I have no evidence to show.")

    by_ch = hits["Channel"].value_counts()
    ch_line = ", ".join(f"{c}: {int(n)}" for c, n in by_ch.items())
    neg = int((hits["Sentiment"].str.lower() == "negative").sum())
    ans = (
        f"**{len(hits)} of {len(df)}** stored records in **{ws_label(ws)}** match \"{label}\" "
        f"({neg} labelled negative). By channel — {ch_line}."
    )
    mems, note = recall_workspace_memories(ws, label)
    return _pi_result(ans, _pi_sample(hits, 8), relevant=len(hits), memories=mems, memory_note=note)


def pi_answer(question: str, ws: str) -> dict:
    """Route a question to a deterministic handler using only this workspace's data."""
    df = _pi_frame(ws_feedback(ws))
    if df.empty:
        return _pi_result(f"There is no stored feedback for **{ws_label(ws)}** yet, so I can't answer.")
    q = question.lower()
    has = lambda *keys: any(k in q for k in keys)  # noqa: E731

    if has("evidence", "example", "proof", "quotes"):
        return _pi_evidence(df, question, ws)
    if has("change", "release", "version"):
        return _pi_changes(df, ws)
    if "channel" in q:
        return _pi_channel(df, ws)
    if has("recurring", "repeat", "keep seeing"):
        return _pi_recurring(df, ws)
    if has("learn", "remember", "memory", "previous", "histor"):
        return _pi_learned(df, ws)
    if has("unhappy", "why", "complain", "negative", "frustrat", "upset", "dissatisf"):
        return _pi_unhappy(df, ws)
    if has(
    "theme",
    "topic",
    "top",
    "main customer problem",
    "main customer problems",
    "biggest problem",
    "biggest problems",
    "major problem",
    "major problems",
    "main issue",
    "main issues",
    "biggest issue",
    "biggest issues",
    "pain point",
    "pain points",
    "customer problem",
    "customer problems",
):
        return _pi_themes(df, ws)
    return _pi_result(
        "I can't answer that from the stored data. I can answer about: top themes, why customers are unhappy, "
        "complaints by channel, recurring issues, what the agent learned, product-change impact, and "
        "\"Show evidence for <topic>\"."
    )


# --- UI ------------------------------------------------------------------
def _pi_render_answer(res: dict):
    st.markdown(res["answer"])
    with st.expander("Evidence"):
        st.markdown("**📄 Data evidence** — exact stored feedback")
        data = res.get("data")
        if data is None or data.empty:
            st.caption("No supporting feedback records for this answer.")
        else:
            if res.get("relevant") and res["relevant"] > len(data):
                st.caption(f"Showing {len(data)} of {res['relevant']} relevant stored records.")
            st.dataframe(data[_PI_COLS], use_container_width=True, hide_index=True)

        st.markdown("**🧠 Memory** — Hindsight-recalled historical experience")
        mems = res.get("memories") or []
        if not mems:
            st.caption(res.get("memory_note") or "Not used for this question.")
        for m in mems:
            with st.container(border=True):
                st.markdown(f"**{m['text']}**")
                st.caption(" · ".join(x for x in (m["timestamp"], m["channel"], m["theme"]) if x))
                if m["issue"]:
                    st.write(f"Underlying issue: {m['issue']}")


def render_product_intelligence_agent(ws: str):
    st.subheader("🧠 Ask the Product Intelligence Agent")
    st.caption(
        f"Answers come only from **{ws_label(ws)}** stored feedback, its product changes and its Hindsight memories. "
        "Counts are computed in Python from stored labels. Also try: \"What changed after a product change?\" "
        "or \"Show evidence for <theme or keyword>\"."
    )

    history = st.session_state.setdefault("pi_chat", {}).setdefault(ws, [])

    cols = st.columns(len(PI_SUGGESTED_QUESTIONS))
    clicked = None
    for i, (c, q) in enumerate(zip(cols, PI_SUGGESTED_QUESTIONS)):
        if c.button(q, key=f"pi_chip_{ws}_{i}", use_container_width=True):
            clicked = q
    if history and st.button("Clear conversation", key=f"pi_clear_{ws}"):
        history.clear()

    typed = st.chat_input("Ask about this workspace's feedback…", key=f"pi_input_{ws}")
    question = clicked or typed
    if question:
        with st.spinner("Checking stored feedback and memories…"):
            history.append({"q": question, "res": pi_answer(question, ws)})

    for turn in history:
        with st.chat_message("user"):
            st.write(turn["q"])
        with st.chat_message("assistant"):
            _pi_render_answer(turn["res"])


# =========================================================================
# Feedback Inbox
# =========================================================================
if page == "📥 Feedback Inbox":
    st.header("📥 Feedback Inbox")
    st.caption(count_line(workspace))

    feedback = ws_feedback(workspace)
    channels = sorted({f.get("channel", "Unknown") for f in feedback})
    tab_labels = ["All"] + channels
    tabs = st.tabs(tab_labels)

    def render_records(records: list[dict]):
        if not records:
            st.info("No records in this channel.")
            return
        rows = []
        for r in records:
            a = r.get("analysis") or {}
            rows.append({
                "Date": r.get("timestamp", ""),
                "Channel": r.get("channel", ""),
                "Feedback": r.get("text", ""),
                "Theme": a.get("theme", ""),
                "Sentiment": a.get("sentiment", ""),
                "Priority": a.get("priority", ""),
                "Lifecycle": r.get("lifecycle", ""),
                "Source": r.get("source_type", ""),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with tabs[0]:
        render_records(feedback)
    for i, ch in enumerate(channels, start=1):
        with tabs[i]:
            render_records([f for f in feedback if f.get("channel") == ch])


# =========================================================================
# Hindsight Memory
# =========================================================================
elif page == "🧠 Hindsight Memory":
    st.header("🧠 Hindsight Memory")
    st.write("**What does the agent remember?**")

    if not memory_is_on(workspace):
        st.warning("Agent Memory is OFF for this workspace.")
    elif not hindsight_ready():
        st.warning("Hindsight is not configured.")
    else:
        # Recall historical memories directly for this workspace.
        memories, memory_note = recall_workspace_memories(
            workspace,
            "historical customer feedback experiences",
            limit=10,
        )

        if memories:
            st.success(
                f"✓ Hindsight recalled {len(memories)} historical "
                f"experience(s) for **{ws_label(workspace)}**."
            )

            st.markdown("### 🧠 Stored historical experiences")

            for m in memories:
                with st.container(border=True):
                    st.markdown(f"**{m['text']}**")

                    details = " · ".join(
                        x for x in (
                            m.get("timestamp"),
                            m.get("channel"),
                            m.get("theme"),
                        )
                        if x
                    )

                    if details:
                        st.caption(details)

                    if m.get("issue"):
                        st.write(
                            f"Underlying issue: {m['issue']}"
                        )
        else:
            st.info(
                f"No Hindsight memories were returned for "
                f"**{ws_label(workspace)}**. {memory_note}"
            )

        # Also show the latest new-feedback reasoning, when available.
        last = last_result_for(workspace)

        if last:
            st.divider()
            st.markdown("### 🔁 Latest memory interaction")

            status = last.get("memory_status")

            if status == "ok" or status == "partial":
                st.success(
                    f"✓ {last.get('memory_message', '')}"
                )
            else:
                st.warning(
                    last.get(
                        "memory_message",
                        "Hindsight was not used.",
                    )
                )

            strong = [
                e for e in last.get("evidence", [])
                if e.is_strong
            ]

            if strong:
                st.markdown("### 🔎 Relevant recalled experiences")

                for e in strong:
                    m = e.memory

                    with st.container(border=True):
                        st.markdown(f"**{m.raw_text}**")
                        st.caption(
                            f"{m.timestamp} · "
                            f"{m.channel} · "
                            f"{m.discovered_theme}"
                        )

                        if m.underlying_issue:
                            st.write(
                                f"Underlying issue: "
                                f"{m.underlying_issue}"
                            )

                        st.write(
                            f"Why relevant: {e.reason}"
                        )

                        st.caption(
                            f"Lifecycle: {m.lifecycle_state}"
                        )

# =========================================================================
# Feedback Time Machine
# =========================================================================
elif page == "🕰️ Feedback Time Machine":
    st.header("🕰️ Feedback Time Machine")
    last = last_result_for(workspace)
    if not last:
        st.info("Submit feedback first to build the memory timeline.")
    else:
        lifecycle = last["lifecycle"].value
        color = {
            "NOVEL": "🆕",
            "RECURRING": "🔁",
            "IMPROVING_RESOLVED": "📈",
            "POSSIBLE_REGRESSION": "🚨",
            "INSUFFICIENT_EVIDENCE": "⚪",
        }.get(lifecycle, "•")
        st.markdown(f"## {color} {lifecycle}")
        st.write(last["reason"])

        strong = [e for e in last["evidence"] if e.is_strong]
        for e in strong:
            m = e.memory
            st.markdown(f"📝 **{m.raw_text}**")
            st.caption(f"{m.timestamp} · {m.channel} · {m.discovered_theme}")
            st.markdown("↓")

        # Product change (if relevant)
        matching_changes = [
            pc for pc in load_product_changes()
            if pc.get("workspace") == workspace
            and (pc.get("linked_theme") == last["analysis"].theme)
        ]
        for pc in matching_changes:
            st.markdown(f"🔧 **Product change — {pc['title']} (v{pc['version']})**")
            st.caption(f"{pc['date']} · {pc['problem_issue']}")
            st.write(f"Expected: {pc['expected_outcome']}")
            if pc.get("observed_outcome"):
                st.write(f"Observed: {pc['observed_outcome']}")
            st.markdown("↓")

        st.markdown(f"🚨 **Current feedback:** {last['item'].text}")
        st.caption(f"{last['item'].timestamp} · {last['item'].channel}")

# =========================================================================
# Product Intelligence
# =========================================================================
elif page == "📊 Product Intelligence":
    st.header("📊 Product Intelligence")
    st.caption(
        f"AI-generated insights from feedback accumulated in the "
        f"**{ws_label(workspace)}** workspace."
    )

    st.toggle(
        f"🧠 Agent Memory: {'ON' if memory_is_on(workspace) else 'OFF'}",
        value=memory_is_on(workspace),
        key=f"memory_toggle_{workspace}",
        on_change=_set_memory_toggle,
        args=(workspace,),
    )
    if memory_is_on(workspace):
        st.caption("Memory ON — historical Hindsight experience may be used.")
    else:
        st.caption("Memory OFF — current feedback only.")

    feedback = ws_feedback(workspace)

    if not feedback:
        st.info("No feedback for this workspace yet.")
        st.divider()
        render_teach_agent(workspace, display_name=ws_label(workspace))
        st.divider()
        render_product_intelligence_agent(workspace)

    else:
        # -------------------------------------------------------------
        # Build analytics
        # -------------------------------------------------------------
        themes = {}
        sentiments = {}
        priorities = {}
        lifecycles = {}
        channels_count = {}

        for f in feedback:
            analysis = f.get("analysis") or {}

            theme = analysis.get("theme") or "Unknown"
            sentiment = analysis.get("sentiment") or "Unknown"
            priority = analysis.get("priority") or "Unknown"
            lifecycle = f.get("lifecycle") or "NOVEL"
            channel = f.get("channel") or "Unknown"

            themes[theme] = themes.get(theme, 0) + 1
            sentiments[sentiment] = sentiments.get(sentiment, 0) + 1
            priorities[priority] = priorities.get(priority, 0) + 1
            lifecycles[lifecycle] = lifecycles.get(lifecycle, 0) + 1
            channels_count[channel] = channels_count.get(channel, 0) + 1

        # -------------------------------------------------------------
        # TOP METRICS
        # -------------------------------------------------------------
        total = len(feedback)

        negative = sum(
            1
            for f in feedback
            if str(
                (f.get("analysis") or {}).get("sentiment", "")
            ).lower() == "negative"
        )

        high_priority = sum(
            1
            for f in feedback
            if str(
                (f.get("analysis") or {}).get("priority", "")
            ).lower() in {"high", "critical"}
        )

        recurring = lifecycles.get("RECURRING", 0)

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("💬 Total Feedback", total)
        c2.metric("📡 Channels", len(channels_count))
        c3.metric("🔴 Negative Signals", negative)
        c4.metric("🔁 Recurring Issues", recurring)

        st.divider()

        # -------------------------------------------------------------
        # FEEDBACK THEMES
        # -------------------------------------------------------------
        st.subheader("🔥 Feedback Themes")

        theme_rows = sorted(
            themes.items(),
            key=lambda x: x[1],
            reverse=True
        )

        theme_df = pd.DataFrame(
            theme_rows,
            columns=["Theme", "Feedback"]
        )

        st.dataframe(
            theme_df,
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        # -------------------------------------------------------------
        # CHANNELS
        # -------------------------------------------------------------
        st.subheader("📡 Feedback Channel Coverage")

        channel_rows = sorted(
            channels_count.items(),
            key=lambda x: x[1],
            reverse=True
        )

        channel_df = pd.DataFrame(
            channel_rows,
            columns=["Channel", "Feedback"]
        )

        channel_df["Share"] = (
            channel_df["Feedback"] / total * 100
        ).round(1).astype(str) + "%"

        st.dataframe(
            channel_df,
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        # -------------------------------------------------------------
        # SENTIMENT + LIFECYCLE
        # -------------------------------------------------------------
        left, right = st.columns(2)

        with left:
            st.subheader("😊 Sentiment")

            sentiment_rows = sorted(
                sentiments.items(),
                key=lambda x: x[1],
                reverse=True
            )

            sentiment_df = pd.DataFrame(
                sentiment_rows,
                columns=["Sentiment", "Count"]
            )

            st.dataframe(
                sentiment_df,
                use_container_width=True,
                hide_index=True
            )

        with right:
            st.subheader("🔁 Lifecycle Signals")

            lifecycle_rows = sorted(
                lifecycles.items(),
                key=lambda x: x[1],
                reverse=True
            )

            lifecycle_df = pd.DataFrame(
                lifecycle_rows,
                columns=["Lifecycle", "Count"]
            )

            st.dataframe(
                lifecycle_df,
                use_container_width=True,
                hide_index=True
            )

        st.divider()

        # -------------------------------------------------------------
        # PRIORITY
        # -------------------------------------------------------------
        st.subheader("🚨 Priority Distribution")

        priority_rows = sorted(
            priorities.items(),
            key=lambda x: x[1],
            reverse=True
        )

        priority_df = pd.DataFrame(
            priority_rows,
            columns=["Priority", "Count"]
        )

        st.dataframe(
            priority_df,
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        # -------------------------------------------------------------
        # AGENT LEARNING SUMMARY
        # -------------------------------------------------------------
        st.subheader("🧠 What the Agent Learned")

        top_theme = (
            max(themes, key=themes.get)
            if themes
            else "No dominant theme"
        )

        top_channel = (
            max(channels_count, key=channels_count.get)
            if channels_count
            else "Unknown"
        )

        if recurring > 0:
            st.success(
                f"**{recurring} recurring issue(s)** detected. "
                f"**{top_theme}** is currently the most frequent "
                f"feedback theme, with **{top_channel}** as the "
                f"largest feedback source."
            )
        else:
            st.info(
                f"**{top_theme}** is currently the most frequent "
                f"feedback theme, with **{top_channel}** as the "
                f"largest feedback source."
            )

        st.caption(
            "Insights are generated from feedback currently stored "
            "in this workspace. No cross-workspace data is used."
        )

        st.divider()
        render_wave_comparison(workspace, feedback, display_name=ws_label(workspace))
        st.divider()
        render_teach_agent(workspace, display_name=ws_label(workspace))
        st.divider()
        render_product_intelligence_agent(workspace)

# =========================================================================
# Live Feedback
# =========================================================================
elif page == "🎤 Live Feedback":
    st.header("🎤 Live Feedback")
    st.caption(count_line(workspace))

    channel = st.selectbox(
        "Channel",
        ["App Reviews", "Support Tickets", "Email", "Surveys", "Voice"],
    )
    text = st.text_area("Feedback text", height=120, key=f"live_text_{workspace}")

    if st.button("Analyze & Remember", type="primary"):
        if not text.strip():
            st.error("Please enter feedback text.")
        else:
            with st.status("Understanding feedback...", expanded=True) as status:
                st.write("Recalling experience...")
                result = process_feedback(workspace, channel, text, source_type="live", use_memory=memory_is_on(workspace))
                if not result.get("ok"):
                    status.update(label="Analysis failed.", state="error")
                    st.error(result.get("error", "Unknown error"))
                else:
                    st.write("Evaluating evidence...")
                    st.write("Checking product history...")
                    st.write("Retaining new experience...")
                    status.update(label="Decision ready.", state="complete")
                    st.session_state["last_result"] = result

            if result.get("ok"):
                st.markdown("### Lifecycle decision")
                st.success(result["lifecycle"].value)
                st.write(result["reason"])

                if result["memory_status"] == "disabled":
                    st.info(result["memory_message"])
                elif result["memory_status"] != "ok":
                    st.warning(result["memory_message"])
                else:
                    st.info(result["memory_message"])

                st.markdown("### Understanding")
                st.json(result["analysis"].to_dict())

                strong = [e for e in result["evidence"] if e.is_strong]
                if strong:
                    st.markdown("### Relevant historical experience")
                    for e in strong:
                        m = e.memory
                        st.markdown(f"- **{m.raw_text}**  \n  _{m.timestamp} · {m.channel} · relevance {e.relevance:.2f}_")
                        st.caption(e.reason)
                else:
                    st.markdown("### No strongly relevant historical experience found.")

                st.caption(f"Count now: {count_line(workspace)}")

    st.divider()
    st.subheader("Record a product change")
    with st.form(f"product_change_form_{workspace}"):
        pc_title = st.text_input("Change title")
        pc_version = st.text_input("Version", value="")
        pc_date = st.text_input("Date (YYYY-MM-DD)")
        pc_issue = st.text_input("Problem addressed")
        pc_expected = st.text_input("Expected outcome")
        pc_observed = st.text_input("Observed outcome (optional)")
        pc_theme = st.text_input("Linked theme (must match an existing theme)")
        pc_cap = st.text_input("Linked capability (optional)")
        submitted = st.form_submit_button("Record product change")
        if submitted:
            if not (pc_title and pc_date and pc_issue and pc_expected):
                st.error("Title, date, problem, and expected outcome are required.")
            else:
                pc = record_product_change(
                    workspace=workspace,
                    title=pc_title,
                    version=pc_version or "n/a",
                    date=pc_date,
                    problem_issue=pc_issue,
                    expected_outcome=pc_expected,
                    observed_outcome=pc_observed,
                    linked_theme=pc_theme,
                    linked_capability=pc_cap,
                )
                st.success(f"Recorded product change: {pc.title}")


# =========================================================================
# Judge Test Mode
# =========================================================================
elif page == "🧪 Judge Test Mode":
    st.header("🧪 Judge Test Mode")

    st.success(f"🟢 Testing workspace: **{ws_label(workspace)}**")

    st.write(
        "Test the Feedback Memory Agent with your own historical "
        "feedback dataset and new customer experiences."
    )

    st.caption(count_line(workspace))

    st.markdown("### 1 · Import historical feedback")
    st.caption(
        "Upload customer feedback with any column layout. CSV, JSON and JSONL are supported. "
        "Columns are detected automatically and you confirm the mapping before analysis."
    )
    uploaded = st.file_uploader(
        "Upload historical feedback",
        type=["csv", "json", "jsonl"],
        key=f"judge_dataset_{workspace}",
    )
    if uploaded is not None:
        raw_bytes = uploaded.getvalue()
        fkey = hashlib.md5(raw_bytes + workspace.encode("utf-8")).hexdigest()[:8]
        table = None
        load_info: dict = {}
        try:
            table, load_info = load_table(uploaded.name, raw_bytes)
        except IngestError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Could not read this file: {e}")

        if table is not None:
            cols = list(table.columns)
            profile = profile_columns(table)
            detected = detect_mapping(table, profile)

            st.caption(
                f"{load_info['rows']} rows · {len(cols)} columns · "
                f"encoding: {load_info['encoding']} · "
                f"{load_info['blank_rows_dropped']} blank rows skipped"
            )
            with st.expander("Column inspection (names and sample values)"):
                st.dataframe(profile_table(profile), use_container_width=True, hide_index=True)

            st.markdown("#### Column mapping")
            st.caption("Detected automatically. Adjust anything that looks wrong.")
            NONE = "(none)"
            opts = [NONE] + cols

            def _pick(label: str, role: str):
                d = detected.get(role)
                return st.selectbox(
                    label, opts,
                    index=opts.index(d) if d in opts else 0,
                    key=f"map_{role}_{fkey}",
                )

            text_cols = st.multiselect(
                "Feedback text column(s) — multiple allowed",
                cols,
                default=detected["text_cols"],
                key=f"map_text_{fkey}",
            )
            mc1, mc2, mc3 = st.columns(3)
            with mc1:
                date_pick = _pick("Date / time", "date_col")
                id_pick = _pick("ID", "id_col")
            with mc2:
                rating_pick = _pick("Rating", "rating_col")
                channel_pick = _pick("Channel / source", "channel_col")
            with mc3:
                segment_pick = _pick("Segment / user / tier", "segment_col")

            mapping = {
                "text_cols": text_cols,
                "date_col": None if date_pick == NONE else date_pick,
                "rating_col": None if rating_pick == NONE else rating_pick,
                "id_col": None if id_pick == NONE else id_pick,
                "channel_col": None if channel_pick == NONE else channel_pick,
                "segment_col": None if segment_pick == NONE else segment_pick,
            }

            problems = validate_mapping(table, mapping)
            if problems:
                for p in problems:
                    st.error(p)
            else:
                norm_records, norm_stats = normalize_table(table, mapping)
                if not norm_records:
                    st.error(
                        "No usable feedback text was found in the selected column(s) — "
                        "every row is empty there. Please choose a different text column."
                    )
                else:
                    norm_summary = summarize_records(norm_records)
                    st.markdown("#### Mapping preview")
                    st.dataframe(
                        mapping_preview(mapping, table),
                        use_container_width=True, hide_index=True,
                    )
                    if not mapping["date_col"]:
                        st.info("No date column selected — the upload will be treated as one wave.")
                    elif not norm_stats["dated"]:
                        st.warning(
                            "No values in the date column could be read as dates — "
                            "the upload will be treated as one wave."
                        )
                    else:
                        st.caption(
                            f"Dates {norm_stats['date_min']} → {norm_stats['date_max']}"
                        )

                    pm1, pm2, pm3, pm4 = st.columns(4)
                    pm1.metric("Records to analyze", norm_summary["total"])
                    pm2.metric("Waves", len(norm_summary["waves"]))
                    pm3.metric("Channels", len(norm_summary["channels"]))
                    pm4.metric("Skipped (no text)", norm_stats["rows_without_text"])
                    if norm_summary["rating_mean"] is not None:
                        st.caption(
                            f"Average rating: {norm_summary['rating_mean']:.2f} "
                            f"({norm_summary['rating_count']} rated records)"
                        )

                    st.markdown("Normalized records (first 5)")
                    st.dataframe(
                        preview_frame(norm_records, 5),
                        use_container_width=True, hide_index=True,
                    )
                    st.info(
                        f"These records will become historical experiences for the "
                        f"**{ws_label(workspace)}** workspace and will be analyzed and remembered by Hindsight."
                    )

                    if st.button("Confirm mapping & analyze", type="primary", key=f"confirm_{fkey}"):
                        progress = st.progress(0.0)
                        status_line = st.empty()

                        def cb(i, total, msg):
                            progress.progress(min(i / max(total, 1), 1.0))
                            status_line.caption(f"{i}/{total} — {msg}")

                        summary = import_normalized(
                            norm_records, workspace, source_type="uploaded", progress_cb=cb
                        )
                        progress.progress(1.0)
                        status_line.empty()

                        st.success(f"✓ {summary['imported']} records imported")
                        st.write(f"✓ {summary['analyzed']} analyzed")
                        st.write(f"✓ {summary['themes']} themes discovered")
                        st.write(f"✓ {summary['retained']} experiences retained in Hindsight")
                        st.markdown("### 📡 Channels detected")
                        for channel_name, channel_count in sorted(norm_summary["channels"].items()):
                            st.write(f"**{channel_name}:** {channel_count}")
                        if len(norm_summary["waves"]) > 1 or "wave-1" not in norm_summary["waves"]:
                            st.markdown("### 🌊 Waves")
                            for wave_name, wave_count in norm_summary["waves"].items():
                                st.write(f"**{wave_name}:** {wave_count}")
                        if summary["failures"]:
                            st.warning(f"Some analyses failed: {summary['failures'][0]}")

    st.divider()

    st.markdown("### 2 · Paste multiple records")
    st.caption("One feedback item per line, or paste CSV/JSON.")
    pasted = st.text_area("Pasted data", height=160, key=f"judge_paste_{workspace}")
    if st.button("Import pasted"):
        lines = [l.strip() for l in pasted.splitlines() if l.strip()]
        if not lines:
            st.error("Nothing to import.")
        else:
            # Try JSON first, then CSV, then plain lines
            records = []
            try:
                maybe_json = json.loads(pasted)
                records = maybe_json if isinstance(maybe_json, list) else [maybe_json]
            except Exception:
                if "," in lines[0]:
                    import csv, io
                    reader = csv.DictReader(io.StringIO(pasted))
                    records = list(reader)
                else:
                    records = [{"text": l} for l in lines]
            summary = import_records(records, workspace, source_type="uploaded")
            st.success(f"✓ {summary['imported']} records imported, "
                       f"{summary['analyzed']} analyzed, "
                       f"{summary['retained']} retained in Hindsight.")

    st.divider()

    st.markdown("### 3 · Test the Agent with New Feedback")

    st.caption(
    f"Add a new customer experience to the **{ws_label(workspace)}** workspace. "
    "The agent will recall historical experience from Hindsight before deciding."
)
    judge_text = st.text_area("New feedback", height=100, key=f"judge_single_{workspace}")
    judge_channel = st.selectbox(
        "Channel", ["App Reviews", "Support Tickets", "Email", "Surveys", "Voice"], key=f"judge_channel_{workspace}"
    )
    if st.button("Analyze & remember", key=f"judge_single_btn_{workspace}"):
        if not judge_text.strip():
            st.error("Enter feedback text.")
        else:
            with st.status("Running agent...", expanded=True) as status:
                st.write("Understanding feedback...")
                result = process_feedback(workspace, judge_channel, judge_text, source_type="live", use_memory=memory_is_on(workspace))
                if not result.get("ok"):
                    status.update(label="Failed.", state="error")
                    st.error(result.get("error"))
                else:
                    st.write("Recalling experience...")
                    st.write("Evaluating evidence...")
                    st.write("Retaining memory...")
                    status.update(label="Done.", state="complete")
                    st.session_state["last_result"] = result
                    st.success(result["lifecycle"].value)
                    st.write(result["reason"])
                    st.caption(result["memory_message"])

    st.divider()

    st.markdown("### 4 · Voice feedback")
    audio = st.file_uploader("Upload audio", type=["wav", "mp3", "m4a", "ogg"], key="judge_audio")
    if audio is not None and st.button("Transcribe & analyze"):
        try:
            transcript = transcribe_audio(audio.read(), filename=audio.name)
        except VoiceUnavailable as e:
            st.error(str(e))
        else:
            st.info(f"Transcript: {transcript}")
            result = process_feedback(workspace, "Voice", transcript, source_type="voice", use_memory=memory_is_on(workspace))
            if not result.get("ok"):
                st.error(result.get("error"))
            else:
                st.session_state["last_result"] = result
                st.success(result["lifecycle"].value)
                st.write(result["reason"])
                st.caption(result["memory_message"])

    st.divider()
    with st.expander("Danger zone"):
        st.caption(
            f"Affects only **{ws_label(workspace)}**: removes its live/imported feedback. "
            "Seed data, other workspaces, taught rules, product changes and Hindsight memories are not touched."
        )
        if st.button("Clear live/imported feedback (keeps seed data)", key=f"clear_live_{workspace}"):
            removed = clear_live_feedback_for_workspace(workspace)
            if last_result_for(workspace):
                st.session_state.pop("last_result", None)
            st.success(f"Cleared {removed} live/imported record(s) from {ws_label(workspace)}. Seed data untouched.")