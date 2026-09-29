"""Streamlit sections: 🌊 Wave Comparison and 🧠 Teach the Agent.

Rendered from the Product Intelligence page. Logic lives in wave_compare.py
and teach.py; this file only draws it.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from storage import workspace_display_name
from teach import load_rules, teach
from wave_compare import (
    MODE_MONTH,
    MODE_WAVE_ID,
    build_wave_frame,
    compare_waves,
    list_waves,
)

_MODES = {
    "Month (from feedback date)": MODE_MONTH,
    "Wave ID (from import)": MODE_WAVE_ID,
}


def _pct(v) -> str:
    return "n/a" if v is None else f"{v:.1f}%"


def _theme_list(title: str, items: list[str]) -> None:
    st.markdown(f"**{title}**")
    if not items:
        st.caption("None")
    for it in items:
        st.markdown(f"- {it}")


def _change_list(title: str, frame: pd.DataFrame) -> None:
    st.markdown(f"**{title}**")
    if frame is None or frame.empty:
        st.caption("None")
        return
    for _, r in frame.iterrows():
        st.markdown(f"- {r['Theme']}: {int(r['Wave A count'])} → {int(r['Wave B count'])} ({int(r['Change']):+d})")


def render_wave_comparison(workspace: str, records: list[dict], display_name: str | None = None) -> None:
    # `workspace` is the internal id (logic/keys); `shown` is UI text only.
    shown = display_name or workspace_display_name(workspace)
    st.subheader("🌊 Wave Comparison")
    st.caption(
        f"Compare two waves of **{shown}** feedback. All numbers are computed in Python "
        "from stored records; themes reflect anything you taught the agent."
    )

    if not records:
        st.info("No feedback for this workspace yet, so there are no waves to compare.")
        return

    mode_label = st.radio(
        "Group waves by", list(_MODES), horizontal=True, key=f"wc_mode_{workspace}"
    )
    mode = _MODES[mode_label]

    df = build_wave_frame(records, load_rules(workspace), mode)
    waves = list_waves(df)

    undated = int(df["wave"].isna().sum())
    if undated:
        st.caption(f"{undated} record(s) have no usable date/wave and are not part of any wave.")

    if not waves:
        st.info("No waves found. Records need a readable date (or a wave ID from an import).")
        return
    if len(waves) == 1:
        n = int((df["wave"] == waves[0]).sum())
        st.info(f"Only one wave exists so far (**{waves[0]}**, {n} records). Import or add feedback from another period to compare.")
        return

    c1, c2 = st.columns(2)
    wave_a = c1.selectbox("Wave A", waves, index=0, key=f"wc_a_{workspace}_{mode}")
    wave_b = c2.selectbox("Wave B", waves, index=len(waves) - 1, key=f"wc_b_{workspace}_{mode}")

    res = compare_waves(df, wave_a, wave_b)
    if res["error"]:
        st.warning(res["error"])
        return

    sa, sb = res["stats_a"], res["stats_b"]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(f"💬 Feedback · {wave_a}", sa["total"])
    m2.metric(f"💬 Feedback · {wave_b}", sb["total"], delta=sb["total"] - sa["total"])
    m3.metric(f"🔴 Negative · {wave_a}", _pct(sa["negative_pct"]), help=f"{sa['negative']} of {sa['total']}")
    delta = None
    if sa["negative_pct"] is not None and sb["negative_pct"] is not None:
        delta = f"{sb['negative_pct'] - sa['negative_pct']:+.1f} pts"
    m4.metric(f"🔴 Negative · {wave_b}", _pct(sb["negative_pct"]), delta=delta,
              delta_color="inverse", help=f"{sb['negative']} of {sb['total']}")

    if sa["total"] != sb["total"]:
        st.caption("The waves have different sizes, so compare the share % columns as well as raw counts.")
    fixed = sa["reclassified"] + sb["reclassified"]
    if fixed:
        st.caption(f"{fixed} record(s) in these waves are counted under a taught theme instead of their stored theme.")
    no_theme = sa["no_theme"] + sb["no_theme"]
    if no_theme:
        st.caption(f"{no_theme} record(s) have no theme and are left out of the theme tables.")

    table = res["table"]
    if table.empty:
        st.info("No themes were recorded in these two waves, so there is no theme comparison.")
        return

    t1, t2 = st.columns(2)
    with t1:
        st.markdown(f"**Top themes · {wave_a}**")
        st.dataframe(res["top_a"], use_container_width=True, hide_index=True)
    with t2:
        st.markdown(f"**Top themes · {wave_b}**")
        st.dataframe(res["top_b"], use_container_width=True, hide_index=True)

    l1, l2 = st.columns(2)
    with l1:
        _change_list("📈 Increased", res["increased"])
        _theme_list("🆕 Newly appearing", res["new"])
    with l2:
        _change_list("📉 Decreased", res["decreased"])
        _theme_list("👋 Disappeared", res["disappeared"])
    if res["unchanged"]:
        st.caption("Unchanged (equal counts): " + ", ".join(res["unchanged"]))

    with st.expander("Full theme comparison"):
        st.dataframe(table, use_container_width=True, hide_index=True)


def render_teach_agent(workspace: str, display_name: str | None = None) -> None:
    # `workspace` is the internal id (logic/keys); `shown` is UI text only.
    shown = display_name or workspace_display_name(workspace)
    st.subheader("🧠 Teach the Agent")
    st.caption(
        f"Correct how **{shown}** feedback is themed. Future feedback (live or imported) that contains "
        "any of these phrases will be classified under your theme. Corrections apply to this workspace only."
    )

    with st.form(f"teach_form_{workspace}", clear_on_submit=True):
        theme = st.text_input("Theme", placeholder="Export Failure")
        phrases = st.text_area(
            "Correction / aliases (separate with commas)",
            placeholder="CSV export, CSV download, export not working, unable to download report",
            height=80,
        )
        submitted = st.form_submit_button("Teach the agent")

    if submitted:
        res = teach(workspace, theme, phrases)
        if not res["ok"]:
            st.error(res["error"])
        else:
            rule = res["rule"]
            st.success(f"Learned: {len(rule['phrases'])} phrase(s) → **{rule['theme']}**")
            if res["hindsight_ok"]:
                st.info(res["hindsight_message"])
            else:
                st.warning(res["hindsight_message"])

    rules = load_rules(workspace)
    st.markdown("**Learned corrections**")
    if not rules:
        st.caption("Nothing taught for this workspace yet.")
        return
    st.dataframe(
        pd.DataFrame([
            {
                "Theme": r.get("theme", ""),
                "Phrases": ", ".join(r.get("phrases") or []),
                "Taught at (UTC)": r.get("created_at", ""),
                "Retained in Hindsight": "Yes" if r.get("hindsight_retained") else "No",
            }
            for r in reversed(rules)
        ]),
        use_container_width=True,
        hide_index=True,
    )