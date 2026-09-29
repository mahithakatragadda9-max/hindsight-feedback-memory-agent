import json
import pandas as pd
import streamlit as st

from config import groq_ready, hindsight_ready
from storage import (
    DATA_DIR,
    load_seed_feedback,
    load_live_feedback,
    load_all_feedback,
    load_product_changes,
    clear_live_feedback,
    detect_feedback_column,
)
from seed_data import generate_seed_if_missing
from pipeline import process_feedback, import_records, record_product_change
from voice import transcribe_audio, VoiceUnavailable
from llm import GroqUnavailable

st.set_page_config(page_title="Feedback Memory Intelligence Agent", layout="wide")

# --- Bootstrap seed data
if not (DATA_DIR / "seed_feedback.jsonl").exists():
    generate_seed_if_missing()

# --- Workspaces
DEMO_WORKSPACES = [
    "PayFlow",
    "ShopEase",
    "LearnFlow",
    "TravelMate",
    "TeamDesk",
]

if "custom_workspaces" not in st.session_state:
    st.session_state["custom_workspaces"] = []

all_workspaces = DEMO_WORKSPACES + st.session_state["custom_workspaces"]


def create_workspace():
    clean_name = st.session_state["new_workspace_name"].strip()

    if not clean_name:
        st.session_state["workspace_message"] = "Enter a workspace name."
        return

    if clean_name in all_workspaces:
        st.session_state["workspace_message"] = (
            f"Workspace '{clean_name}' already exists."
        )
        return

    st.session_state["custom_workspaces"].append(clean_name)

    # This is allowed because it happens inside the button callback.
    st.session_state["active_workspace"] = clean_name

    st.session_state["workspace_message"] = (
        f"Created workspace: {clean_name}"
    )


# --- Sidebar
st.sidebar.title("🧠 Feedback Memory Agent")

st.sidebar.markdown("### Workspace")

workspace = st.sidebar.selectbox(
    "Select workspace",
    all_workspaces,
    key="active_workspace",
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


def count_line(ws: str) -> str:
    seed = [f for f in load_seed_feedback() if f.get("workspace") == ws]
    live = [f for f in load_live_feedback() if f.get("workspace") == ws]
    return f"Seed: {len(seed)} | Live/Imported: {len(live)} | Total: {len(seed) + len(live)}"


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

    last = st.session_state.get("last_result")
    if not last:
        st.info("Submit feedback in Live Feedback or Judge Test Mode to see recalled memories.")
    else:
        status = last.get("memory_status")
        if status == "ok" or status == "partial":
            st.success(f"✓ Hindsight memory used — {last.get('memory_message','')}")
        else:
            st.warning(last.get("memory_message", "⚠ Hindsight unavailable."))

        st.markdown("### Recalled experiences")
        strong = [e for e in last["evidence"] if e.is_strong]
        if not strong:
            st.info("No strongly relevant historical experiences found.")
        for e in strong:
            m = e.memory
            with st.container(border=True):
                st.markdown(f"**{m.raw_text}**")
                st.caption(f"{m.timestamp} · {m.channel} · {m.discovered_theme}")
                st.write(f"Underlying issue: {m.underlying_issue}")
                st.write(f"Why relevant: {e.reason}")
                st.write(f"Lifecycle: {m.lifecycle_state}")


# =========================================================================
# Feedback Time Machine
# =========================================================================
elif page == "🕰️ Feedback Time Machine":
    st.header("🕰️ Feedback Time Machine")
    last = st.session_state.get("last_result")
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
        f"**{workspace}** workspace."
    )

    feedback = ws_feedback(workspace)

    if not feedback:
        st.info("No feedback for this workspace yet.")

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
    text = st.text_area("Feedback text", height=120)

    if st.button("Analyze & Remember", type="primary"):
        if not text.strip():
            st.error("Please enter feedback text.")
        else:
            with st.status("Understanding feedback...", expanded=True) as status:
                st.write("Recalling experience...")
                result = process_feedback(workspace, channel, text, source_type="live")
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

                if result["memory_status"] != "ok":
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
    with st.form("product_change_form"):
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

    st.success(f"🟢 Testing workspace: **{workspace}**")

    st.write(
        "Test the Feedback Memory Agent with your own historical "
        "feedback dataset and new customer experiences."
    )

    st.caption(count_line(workspace))

    st.markdown("### 1 · Import historical feedback")
    st.caption(
    "Upload customer feedback from any domain. "
    "CSV, JSON and JSONL are supported."
    )
    uploaded = st.file_uploader(
    "Upload historical feedback",
    type=["csv", "json", "jsonl"],
    key="judge_dataset",
    )
    if uploaded is not None:
        records: list[dict] = []
        try:
            if uploaded.name.lower().endswith(".csv"):
                df = pd.read_csv(uploaded)
                records = df.to_dict(orient="records")
            elif uploaded.name.lower().endswith(".json"):
                data = json.load(uploaded)
                if isinstance(data, dict):
                    data = [data]
                records = data
            elif uploaded.name.lower().endswith(".jsonl"):
                raw = uploaded.read().decode("utf-8", errors="ignore")
                records = [json.loads(line) for line in raw.splitlines() if line.strip()]
            else:
                st.error("Unsupported file type.")
        except Exception as e:
            st.error(f"Could not parse file: {e}")
            records = []

        if records:
            cols = list({k for r in records for k in r.keys()})
            suggested = detect_feedback_column(cols)
            feedback_col = st.selectbox(
                "Which column contains the feedback text?",
                cols,
                index=cols.index(suggested) if suggested in cols else 0,
            )
            st.write(f"Preview of {len(records)} records")
            st.info(
             f"These records will become historical experiences for the "
             f"**{workspace}** workspace and will be analyzed and remembered by Hindsight."
             )

            if st.button("Import & analyze", type="primary"):
                # Map the chosen column into the standardized "text" key
                mapped = []
                for r in records:
                    rr = dict(r)
                    if feedback_col in rr:
                        rr["text"] = rr[feedback_col]
                    mapped.append(rr)

                progress = st.progress(0.0)
                status_line = st.empty()

                def cb(i, total, msg):
                    progress.progress(min(i / max(total, 1), 1.0))
                    status_line.caption(f"{i}/{total} — {msg}")

                summary = import_records(mapped, workspace, source_type="uploaded", progress_cb=cb)
                progress.progress(1.0)
                status_line.empty()

                st.success(f"✓ {summary['imported']} records imported")
                st.write(f"✓ {summary['analyzed']} analyzed")
                st.write(f"✓ {summary['themes']} themes discovered")
                st.write(f"✓ {summary['retained']} experiences retained in Hindsight")
                channel_counts = {}
                for r in mapped:
                    channel = str(r.get("channel") or r.get("source") or "Imported")
                    channel_counts[channel] = channel_counts.get(channel, 0) + 1
                st.markdown("### 📡 Channels detected")
                for channel_name, channel_count in sorted(channel_counts.items()):
                    st.write(f"**{channel_name}:** {channel_count}")
                if summary["failures"]:
                    st.warning(f"Some analyses failed: {summary['failures'][0]}")

    st.divider()

    st.markdown("### 2 · Paste multiple records")
    st.caption("One feedback item per line, or paste CSV/JSON.")
    pasted = st.text_area("Pasted data", height=160)
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
    f"Add a new customer experience to the **{workspace}** workspace. "
    "The agent will recall historical experience from Hindsight before deciding."
)
    judge_text = st.text_area("New feedback", height=100, key="judge_single")
    judge_channel = st.selectbox(
        "Channel", ["App Reviews", "Support Tickets", "Email", "Surveys", "Voice"], key="judge_channel"
    )
    if st.button("Analyze & remember", key="judge_single_btn"):
        if not judge_text.strip():
            st.error("Enter feedback text.")
        else:
            with st.status("Running agent...", expanded=True) as status:
                st.write("Understanding feedback...")
                result = process_feedback(workspace, judge_channel, judge_text, source_type="live")
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
            result = process_feedback(workspace, "Voice", transcript, source_type="voice")
            if not result.get("ok"):
                st.error(result.get("error"))
            else:
                st.session_state["last_result"] = result
                st.success(result["lifecycle"].value)
                st.write(result["reason"])
                st.caption(result["memory_message"])

    st.divider()
    with st.expander("Danger zone"):
        if st.button("Clear live/imported feedback (keeps seed data)"):
            clear_live_feedback()
            st.success("Live feedback cleared. Seed data untouched.")
