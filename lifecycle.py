from datetime import datetime
from models import LifecycleState, ProductChange


def _parse(ts: str) -> datetime | None:
    if not ts:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(ts[: len(fmt.replace("%Y", "0000").replace("%m", "00").replace("%d", "00").replace("%H", "00").replace("%M", "00").replace("%S", "00"))], fmt)
        except Exception:
            continue
    try:
        return datetime.fromisoformat(ts.replace("Z", ""))
    except Exception:
        return None


def _same_theme(a: str, b: str) -> bool:
    if not a or not b:
        return False
    a_n = a.strip().lower()
    b_n = b.strip().lower()
    if a_n == b_n:
        return True
    return a_n in b_n or b_n in a_n


def count_theme_in_window(
    all_records: list[dict],
    theme: str,
    start: datetime | None,
    end: datetime | None,
) -> int:
    count = 0
    for r in all_records:
        analysis = r.get("analysis") or {}
        if not _same_theme(analysis.get("theme", ""), theme):
            continue
        ts = _parse(r.get("timestamp", ""))
        if ts is None:
            continue
        if start and ts < start:
            continue
        if end and ts >= end:
            continue
        count += 1
    return count


def classify_lifecycle(
    analysis,
    evidence,
    product_changes: list[ProductChange],
    all_records: list[dict],
    current_feedback_id: str,
) -> tuple[LifecycleState, str, dict]:
    strong = [e for e in evidence if e.is_strong and not e.is_current_session]

    if not strong:
        return (
            LifecycleState.NOVEL,
            "No sufficiently similar historical experience.",
            {"strong_matches": 0},
        )

    matching_change = None
    for pc in product_changes:
        if _same_theme(pc.linked_theme, analysis.theme) or _same_theme(
            pc.linked_capability, analysis.affected_capability
        ):
            matching_change = pc
            break

    if not matching_change:
        return (
            LifecycleState.RECURRING,
            f"{len(strong)} strongly related historical experience(s) found, "
            "but no completed product change is recorded for this issue.",
            {"strong_matches": len(strong)},
        )

    change_dt = _parse(matching_change.date)
    if change_dt is None:
        return (
            LifecycleState.INSUFFICIENT_EVIDENCE,
            "A product change exists but its date could not be parsed.",
            {"strong_matches": len(strong)},
        )

    # Improvement signal: volume of same-theme feedback before vs shortly after the change.
    pre_volume = count_theme_in_window(
        all_records, analysis.theme, None, change_dt
    )
    from datetime import timedelta
    post_window = count_theme_in_window(
        all_records, analysis.theme, change_dt, change_dt + timedelta(days=21)
    )
    improvement = pre_volume > 0 and post_window <= max(1, pre_volume // 3)

    pre_change_strong = [e for e in strong if (_parse(e.memory.timestamp) or change_dt) < change_dt]
    post_change_strong_prior = [
        e for e in strong
        if (_parse(e.memory.timestamp) or change_dt) > change_dt
    ]

    if not pre_change_strong:
        return (
            LifecycleState.INSUFFICIENT_EVIDENCE,
            "A product change is on file, but no strongly related experience "
            "predates it. Cannot attribute the change to this issue.",
            {"strong_matches": len(strong), "pre_change": 0},
        )
    if post_change_strong_prior:
        return (
            LifecycleState.RECURRING,
            "Related feedback already appeared after the product change, so this "
            "is an ongoing recurrence rather than a fresh regression.",
            {"strong_matches": len(strong)},
        )

    if not improvement:
        return (
            LifecycleState.INSUFFICIENT_EVIDENCE,
            "A product change exists for this issue, but no improvement signal "
            "was observed in the window after the change.",
            {"strong_matches": len(strong), "pre": pre_volume, "post_window": post_window},
        )



    return (
        LifecycleState.POSSIBLE_REGRESSION,
        "A strongly related issue existed, a product change addressed it, "
        "feedback volume dropped afterwards, and now a strongly related "
        "feedback item has appeared post-change.",
        {
            "strong_matches": len(strong),
            "product_change": matching_change.title,
            "pre_change_volume": pre_volume,
            "post_change_window": post_window,
        },
    )
