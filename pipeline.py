import uuid
from datetime import datetime

from models import FeedbackItem, Memory, ProductChange, LifecycleState
from storage import (
    save_live_feedback,
    load_product_changes,
    load_all_feedback,
    append_product_change,
)
from llm import analyze_feedback, evaluate_evidence_llm, GroqUnavailable
from hindsight_client import HindsightClient, HindsightUnavailable
from relevance import build_evidence
from lifecycle import classify_lifecycle


_hindsight = HindsightClient()


def _memory_from_item(item: FeedbackItem, lifecycle_state: str, reason: str) -> Memory:
    a = item.analysis
    return Memory(
        memory_id=str(uuid.uuid4()),
        feedback_id=item.feedback_id,
        workspace=item.workspace,
        timestamp=item.timestamp,
        channel=item.channel,
        raw_text=item.text,
        discovered_theme=a.theme if a else "",
        underlying_issue=a.underlying_issue if a else "",
        sentiment=a.sentiment if a else "unknown",
        priority=a.priority if a else "medium",
        agent_interpretation=reason,
        affected_capability=a.affected_capability if a else "",
        failure_mode=a.failure_mode if a else "",
        lifecycle_state=lifecycle_state,
    )


def process_feedback(
    workspace: str,
    channel: str,
    text: str,
    source_type: str = "live",
) -> dict:
    result: dict = {
        "ok": False,
        "stage": "init",
        "memory_status": "not_attempted",
        "memory_message": "",
        "hindsight_used": False,
    }

    if not text or not text.strip():
        result["error"] = "Feedback text is empty."
        return result

    item = FeedbackItem.new(workspace=workspace, channel=channel, text=text, source_type=source_type)

    # 1. Analysis
    result["stage"] = "analysis"
    try:
        item.analysis = analyze_feedback(text, workspace, channel)
    except GroqUnavailable as e:
        result["error"] = str(e)
        return result

    analysis = item.analysis

    # 2. Recall from Hindsight
    result["stage"] = "recall"
    memories: list[Memory] = []
    try:
        memories = _hindsight.recall(
            query=f"{analysis.underlying_issue} | {analysis.theme} | {analysis.affected_capability}",
            workspace=workspace,
            top_k=20,
        )
        result["memory_status"] = "ok"
        result["hindsight_used"] = True
        result["memory_message"] = f"Recalled {len(memories)} candidate memories from Hindsight."
    except HindsightUnavailable as e:
        result["memory_status"] = "unavailable"
        result["memory_message"] = (
            "⚠ Hindsight unavailable — feedback was analyzed, but persistent "
            f"memory could not be used. ({e})"
        )

    # 3. Evidence evaluation
    result["stage"] = "evidence"
    product_changes = [
        ProductChange(**pc) for pc in load_product_changes()
        if pc.get("workspace") == workspace
    ]

    evaluations: list[dict] = []
    llm_matched_change_id: str | None = None
    if memories or product_changes:
        try:
            ev = evaluate_evidence_llm(
                new_analysis=analysis.to_dict(),
                memories=[
                    {
                        "memory_id": m.memory_id,
                        "timestamp": m.timestamp,
                        "raw_text": m.raw_text,
                        "discovered_theme": m.discovered_theme,
                        "underlying_issue": m.underlying_issue,
                        "affected_capability": m.affected_capability,
                        "failure_mode": m.failure_mode,
                    }
                    for m in memories
                ],
                product_changes=[
                    {
                        "change_id": pc.change_id,
                        "title": pc.title,
                        "version": pc.version,
                        "date": pc.date,
                        "problem_issue": pc.problem_issue,
                        "linked_theme": pc.linked_theme,
                        "linked_capability": pc.linked_capability,
                    }
                    for pc in product_changes
                ],
            )
            evaluations = ev.get("evaluations", []) or []
            llm_matched_change_id = ev.get("matched_product_change_id")
        except GroqUnavailable as e:
            result["memory_message"] += f" | Evidence LLM fallback: {e}"

    evidence = build_evidence(analysis, memories, evaluations, item.feedback_id)

    # 4. Lifecycle
    result["stage"] = "lifecycle"
    all_records = load_all_feedback()

    # Prefer the LLM's product-change match when available.
    if llm_matched_change_id:
        for pc in product_changes:
            if pc.change_id == llm_matched_change_id:
                # put it first so the lifecycle engine picks it
                product_changes.remove(pc)
                product_changes.insert(0, pc)
                break

    lifecycle, reason, details = classify_lifecycle(
        analysis=analysis,
        evidence=evidence,
        product_changes=product_changes,
        all_records=all_records,
        current_feedback_id=item.feedback_id,
    )
    item.lifecycle = lifecycle.value

    # 5. Persist live feedback
    result["stage"] = "persist"
    save_live_feedback(item.to_record())

    # 6. Retain in Hindsight
    result["stage"] = "retain"
    memory = _memory_from_item(item, lifecycle.value, reason)
    try:
        _hindsight.retain(memory)
        if result["memory_status"] == "ok":
            result["memory_message"] += " New experience retained in Hindsight."
    except HindsightUnavailable as e:
        if result["memory_status"] == "ok":
            result["memory_status"] = "partial"
        result["memory_message"] += f" Retain skipped: {e}"

    result["ok"] = True
    result["item"] = item
    result["analysis"] = analysis
    result["evidence"] = evidence
    result["lifecycle"] = lifecycle
    result["reason"] = reason
    result["details"] = details
    return result


def record_product_change(
    workspace: str,
    title: str,
    version: str,
    date: str,
    problem_issue: str,
    expected_outcome: str,
    observed_outcome: str = "",
    linked_theme: str = "",
    linked_capability: str = "",
) -> ProductChange:
    pc = ProductChange(
        change_id=str(uuid.uuid4()),
        workspace=workspace,
        title=title,
        version=version,
        date=date,
        problem_issue=problem_issue,
        expected_outcome=expected_outcome,
        observed_outcome=observed_outcome,
        linked_theme=linked_theme,
        linked_capability=linked_capability,
    )
    append_product_change(pc.to_dict())
    return pc


def import_records(
    records: list[dict],
    workspace: str,
    source_type: str = "uploaded",
    progress_cb=None,
) -> dict:
    """Analyze and retain a batch of imported records.

    Returns summary counts. Records where analysis fails are still stored
    with an empty analysis so the dashboard count is accurate.
    """
    from storage import normalize_imported_record, save_live_feedback

    imported = 0
    analyzed = 0
    themes: set[str] = set()
    retained = 0
    failures: list[str] = []

    total = len(records)
    for i, row in enumerate(records):
        normalized = normalize_imported_record(row, workspace, source_type)
        if normalized is None:
            if progress_cb:
                progress_cb(i + 1, total, "skipped (no feedback text)")
            continue
        imported += 1

        try:
            analysis = analyze_feedback(normalized["text"], workspace, normalized["channel"])
            normalized["analysis"] = analysis.to_dict()
            themes.add(analysis.theme)
            analyzed += 1
        except GroqUnavailable as e:
            failures.append(str(e))
            normalized["analysis"] = None

        save_live_feedback(normalized)

        if normalized["analysis"]:
            mem = Memory(
                memory_id=str(uuid.uuid4()),
                feedback_id=normalized["feedback_id"],
                workspace=workspace,
                timestamp=normalized["timestamp"] or "",
                channel=normalized["channel"],
                raw_text=normalized["text"],
                discovered_theme=normalized["analysis"].get("theme", ""),
                underlying_issue=normalized["analysis"].get("underlying_issue", ""),
                sentiment=normalized["analysis"].get("sentiment", ""),
                priority=normalized["analysis"].get("priority", ""),
                agent_interpretation="Imported historical experience.",
                affected_capability=normalized["analysis"].get("affected_capability", ""),
                failure_mode=normalized["analysis"].get("failure_mode", ""),
                lifecycle_state=LifecycleState.NOVEL.value,
            )
            try:
                _hindsight.retain(mem)
                retained += 1
            except HindsightUnavailable:
                pass

        if progress_cb:
            progress_cb(i + 1, total, normalized["text"][:60])

    return {
        "imported": imported,
        "analyzed": analyzed,
        "retained": retained,
        "themes": len(themes),
        "failures": failures[:3],
    }
