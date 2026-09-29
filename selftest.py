"""Offline self-test of the memory loop and lifecycle engine.

Run:  python selftest.py
Does not require Groq or Hindsight credentials.
"""
import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import storage
from models import FeedbackAnalysis, ProductChange, Memory, LifecycleState
from relevance import Evidence, build_evidence
from lifecycle import classify_lifecycle


def make_record(fid, ts, text, theme, workspace="PayFlow", analysis=None):
    return {
        "feedback_id": fid,
        "workspace": workspace,
        "channel": "Support Tickets",
        "timestamp": ts,
        "text": text,
        "source_type": "test",
        "analysis": analysis or {
            "summary": text, "underlying_issue": text, "theme": theme,
            "sentiment": "negative", "priority": "high",
            "affected_capability": theme, "requested_need": "",
            "recommended_action": "", "failure_mode": text, "confidence": 0.9,
        },
        "lifecycle": "NOVEL",
    }


def run():
    start = datetime(2026, 9, 1)

    # --- Build a small synthetic history
    records = [
        make_record("f1", (start + timedelta(days=1)).isoformat(),
                    "Filtered reports cannot be exported.", "Export Failure"),
        make_record("f2", (start + timedelta(days=3)).isoformat(),
                    "Export button disappears after filtering.", "Export Failure"),
        make_record("f3", (start + timedelta(days=5)).isoformat(),
                    "Filtered export fails.", "Export Failure"),
        # post-change improvement (much fewer)
        make_record("f4", (start + timedelta(days=14)).isoformat(),
                    "Export works now.", "Export Failure"),
        # a totally different issue
        make_record("f5", (start + timedelta(days=2)).isoformat(),
                    "My payment is pending for three hours.", "Payment Pending"),
    ]

    storage.load_all_feedback = lambda: records  # patch for test

    changes = [
        ProductChange(
            change_id="pc1",
            workspace="PayFlow",
            title="Export pipeline reliability update",
            version="2.1",
            date=(start + timedelta(days=11)).isoformat(),
            problem_issue="Filtered export failures",
            expected_outcome="Filtered exports succeed.",
            observed_outcome="Export-related feedback dropped after release.",
            linked_theme="Export Failure",
            linked_capability="Reporting",
        )
    ]

    def evidence_for(new_text, theme, capability, matches):
        analysis = FeedbackAnalysis(
            summary=new_text,
            underlying_issue=new_text,
            theme=theme,
            sentiment="negative",
            priority="high",
            affected_capability=capability,
            requested_need="",
            recommended_action="",
            failure_mode=new_text,
            confidence=0.9,
        )
        mems = []
        evals = []
        for i, (mid, sim) in enumerate(matches):
            src = next(r for r in records if r["feedback_id"] == mid)
            mems.append(Memory(
                memory_id=f"m{mid}",
                feedback_id=src["feedback_id"],
                workspace=src["workspace"],
                timestamp=src["timestamp"],
                channel=src["channel"],
                raw_text=src["text"],
                discovered_theme=src["analysis"]["theme"],
                underlying_issue=src["analysis"]["underlying_issue"],
                sentiment=src["analysis"]["sentiment"],
                priority=src["analysis"]["priority"],
                agent_interpretation="",
            ))
            evals.append({
                "memory_id": f"m{mid}",
                "relevance": sim,
                "same_issue": sim >= 0.75,
                "reason": f"similarity {sim}",
            })
        return analysis, build_evidence(analysis, mems, evals, "current"), changes

    # ---- Test 1: NOVEL
    analysis, evidence, ch = evidence_for(
        "The app is stuck at the login screen.",
        "Login Loop", "Authentication", []
    )
    state, reason, _ = classify_lifecycle(analysis, evidence, ch, records, "current")
    assert state == LifecycleState.NOVEL, f"Test 1 failed: {state}"
    print("✓ Test 1 (NOVEL) passed")

    # ---- Test 2: RECURRING (no product change for this theme)
    records.append(make_record("f6", (start + timedelta(days=4)).isoformat(),
                               "Payment is stuck pending.", "Payment Pending"))
    analysis, evidence, ch = evidence_for(
        "My payment is still pending after hours.",
        "Payment Pending", "Billing",
        [("f5", 0.9), ("f6", 0.85)],
    )
    state, reason, _ = classify_lifecycle(analysis, evidence, ch, records, "current")
    assert state == LifecycleState.RECURRING, f"Test 2 failed: {state}"
    print("✓ Test 2 (RECURRING) passed")

    # ---- Test 3: strong match, change on file, pre-change + post-change recurrence => RECURRING
    # Add an extra post-change recurrence so it's no longer a fresh regression
    records.append(make_record("f7", (start + timedelta(days=20)).isoformat(),
                               "Export fails again after filtering.", "Export Failure"))
    analysis, evidence, ch = evidence_for(
        "Filtered exports still broken today.",
        "Export Failure", "Reporting",
        [("f1", 0.9), ("f2", 0.88), ("f3", 0.9), ("f7", 0.92)],
    )
    state, reason, _ = classify_lifecycle(analysis, evidence, ch, records, "current")
    assert state == LifecycleState.RECURRING, f"Test 3 failed: {state}"
    print("✓ Test 3 (RECURRING after prior recurrence) passed")

    # ---- Test 4: POSSIBLE_REGRESSION
    # Remove the extra post-change recurrence so this is the first one.
    records = [r for r in records if r["feedback_id"] != "f7"]
    analysis, evidence, ch = evidence_for(
        "Filtered exports are failing again after the update.",
        "Export Failure", "Reporting",
        [("f1", 0.9), ("f2", 0.88), ("f3", 0.9)],
    )
    state, reason, _ = classify_lifecycle(analysis, evidence, ch, records, "current")
    assert state == LifecycleState.POSSIBLE_REGRESSION, f"Test 4 failed: {state}"
    print("✓ Test 4 (POSSIBLE_REGRESSION) passed")

    # ---- Test 5: INSUFFICIENT_EVIDENCE (change on file, no improvement signal)
    dense_records = list(records)
    for i in range(6):
        dense_records.append(make_record(
            f"post-{i}", (start + timedelta(days=12 + i)).isoformat(),
            "Export still fails.", "Export Failure",
        ))
    analysis, evidence, ch = evidence_for(
        "Filtered exports are failing again.",
        "Export Failure", "Reporting",
        [("f1", 0.9), ("f2", 0.88)],
    )
    state, reason, _ = classify_lifecycle(analysis, evidence, ch, dense_records, "current")
    assert state in (LifecycleState.INSUFFICIENT_EVIDENCE, LifecycleState.RECURRING), \
        f"Test 5 failed: {state}"
    print("✓ Test 5 (no improvement signal -> not regression) passed")

    print("\nAll self-tests passed.")


if __name__ == "__main__":
    run()
