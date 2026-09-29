import json
import random
from datetime import datetime, timedelta
from pathlib import Path

from storage import DATA_DIR, SEED_FILE, CHANGES_FILE, ensure_data_dir

random.seed(7)

CHANNELS = ["App Reviews", "Support Tickets", "Email", "Surveys", "Voice"]

# Each tuple: (theme, template texts, capability, sentiment bias)
PROBLEMS = [
    (
        "Export Failure",
        [
            "The export button disappears after I apply filters.",
            "Filtered reports cannot be exported at all.",
            "Exporting a filtered report fails silently.",
            "I click export and nothing happens after filtering.",
        ],
        "Reporting",
        "negative",
    ),
    (
        "Auth Logout",
        [
            "The app keeps logging me out every few minutes.",
            "I get signed out unexpectedly while working.",
            "Session expires too aggressively on mobile.",
        ],
        "Authentication",
        "negative",
    ),
    (
        "Search Relevance",
        [
            "Search results are irrelevant to what I typed.",
            "Search ignores my filters and shows old items.",
            "Search ranking is confusing.",
        ],
        "Search",
        "negative",
    ),
    (
        "Delivery Tracking",
        [
            "My delivery tracking has not updated in two days.",
            "The delivery tracker is stuck on 'in transit'.",
            "I cannot see where my order is.",
        ],
        "Delivery",
        "negative",
    ),
    (
        "Dashboard Performance",
        [
            "The new dashboard takes too long to load.",
            "Dashboard is slow on first open.",
            "Charts take several seconds to appear.",
        ],
        "Performance",
        "negative",
    ),
    (
        "Notification Delay",
        [
            "Notifications arrive hours after the event.",
            "Push notifications are delayed.",
            "I miss updates because alerts are late.",
        ],
        "Notifications",
        "negative",
    ),
    (
        "Billing Duplicate",
        [
            "I was charged twice for the same transaction.",
            "Duplicate charge appeared on my account.",
            "Same payment was taken twice.",
        ],
        "Billing",
        "negative",
    ),
    (
        "UI Confusion",
        [
            "The new interface is confusing to navigate.",
            "I cannot find the settings after the redesign.",
            "The layout changed and nothing is where it used to be.",
        ],
        "User Interface",
        "negative",
    ),
    (
        "Report Scheduling",
        [
            "Scheduled reports stopped going out.",
            "My weekly report never arrived.",
            "Scheduled email digest is missing.",
        ],
        "Reporting",
        "negative",
    ),
    (
        "Onboarding",
        [
            "Onboarding flow is too long.",
            "Signup asks for too much information.",
            "I gave up during setup.",
        ],
        "Onboarding",
        "negative",
    ),
    (
        "Positive UX",
        [
            "I love the new design, it feels much cleaner.",
            "Great improvement to the dashboard.",
            "Search feels faster now.",
        ],
        "User Interface",
        "positive",
    ),
]

WORKSPACE_WEIGHTS = {
    "PayFlow": 0.35,
    "ShopEase": 0.2,
    "LearnFlow": 0.15,
    "TravelMate": 0.15,
    "TeamDesk": 0.15,
}


def _pick_workspace() -> str:
    r = random.random()
    acc = 0.0
    for ws, w in WORKSPACE_WEIGHTS.items():
        acc += w
        if r <= acc:
            return ws
    return "PayFlow"


def generate_seed_if_missing(force: bool = False) -> int:
    ensure_data_dir()
    if SEED_FILE.exists() and not force:
        return sum(1 for _ in SEED_FILE.open("r", encoding="utf-8"))

    start = datetime(2026, 9, 1)
    records: list[dict] = []

    # Deliberate longitudinal story for export failures on PayFlow.
    story = [
        # (day_offset, channel, text)
        (1, "Support Tickets", "Filtered reports cannot be exported at all."),
        (2, "App Reviews", "The export button disappears after I apply filters."),
        (4, "Email", "Export fails when I use the date filter."),
        (6, "Surveys", "Filtered export does not work."),
        (9, "Support Tickets", "Still cannot export filtered reports."),
    ]
    for offset, channel, text in story:
        day = start + timedelta(days=offset)
        records.append({
            "feedback_id": f"seed-story-{offset:02d}",
            "workspace": "PayFlow",
            "channel": channel,
            "timestamp": day.isoformat(),
            "text": text,
            "source_type": "synthetic_demo",
            "persona": None,
            "product_version": None,
            "analysis": {
                "summary": text,
                "underlying_issue": "Filtered report export fails.",
                "theme": "Export Failure",
                "sentiment": "negative",
                "priority": "high",
                "affected_capability": "Reporting",
                "requested_need": "Be able to export filtered reports reliably.",
                "recommended_action": "Investigate export pipeline for filtered views.",
                "failure_mode": "Export fails or button disappears after applying filters.",
                "confidence": 0.9,
            },
            "lifecycle": "NOVEL",
        })

    # Post-change improvement window for export failures (fewer records).
    records.append({
        "feedback_id": "seed-story-14",
        "workspace": "PayFlow",
        "channel": "App Reviews",
        "timestamp": (start + timedelta(days=14)).isoformat(),
        "text": "Export seems to work now after the update.",
        "source_type": "synthetic_demo",
        "analysis": {
            "summary": "Export appears fixed.",
            "underlying_issue": "Filtered report export fails.",
            "theme": "Export Failure",
            "sentiment": "positive",
            "priority": "low",
            "affected_capability": "Reporting",
            "requested_need": "Reliable export.",
            "recommended_action": "Monitor for recurrence.",
            "failure_mode": "",
            "confidence": 0.85,
        },
        "lifecycle": "IMPROVING_RESOLVED",
    })

    # Fill out to ~60 with realistic varied records.
    target = 60 - len(records)
    used_templates: list[tuple] = []
    for _ in range(target):
        ws = _pick_workspace()
        theme, templates, capability, sentiment_bias = random.choice(PROBLEMS)
        text = random.choice(templates)
        used_templates.append((ws, theme, text))
        day = start + timedelta(days=random.randint(0, 27))
        channel = random.choice(CHANNELS)
        priority = random.choice(["low", "medium", "high", "critical"]) if sentiment_bias == "negative" else "low"
        sentiment = sentiment_bias if random.random() < 0.75 else "neutral"
        records.append({
            "feedback_id": f"seed-{len(records)+1:03d}",
            "workspace": ws,
            "channel": channel,
            "timestamp": day.isoformat(),
            "text": text,
            "source_type": "synthetic_demo",
            "persona": None,
            "product_version": None,
            "analysis": {
                "summary": text,
                "underlying_issue": text,
                "theme": theme,
                "sentiment": sentiment,
                "priority": priority,
                "affected_capability": capability,
                "requested_need": "Improve this experience.",
                "recommended_action": "Investigate and prioritize.",
                "failure_mode": text if sentiment_bias == "negative" else "",
                "confidence": 0.8,
            },
            "lifecycle": "NOVEL",
        })

    with SEED_FILE.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Seed a demo product change aligned with the export story.
    if not CHANGES_FILE.exists():
        with CHANGES_FILE.open("w", encoding="utf-8") as f:
            demo_change = {
                "change_id": "pc-demo-export-2.1",
                "workspace": "PayFlow",
                "title": "Export pipeline reliability update",
                "version": "2.1",
                "date": (start + timedelta(days=11)).isoformat(),
                "problem_issue": "Filtered report export failures.",
                "expected_outcome": "Filtered reports should export successfully.",
                "observed_outcome": "Export-related feedback volume dropped after release.",
                "linked_theme": "Export Failure",
                "linked_capability": "Reporting",
            }
            f.write(json.dumps(demo_change, ensure_ascii=False) + "\n")

    return len(records)


if __name__ == "__main__":
    n = generate_seed_if_missing(force=True)
    print(f"Wrote {n} synthetic seed records and demo product changes.")
