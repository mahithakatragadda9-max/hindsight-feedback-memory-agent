from __future__ import annotations
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Optional, Any
import uuid


class LifecycleState(str, Enum):
    NOVEL = "NOVEL"
    RECURRING = "RECURRING"
    IMPROVING_RESOLVED = "IMPROVING_RESOLVED"
    POSSIBLE_REGRESSION = "POSSIBLE_REGRESSION"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


@dataclass
class FeedbackAnalysis:
    summary: str = ""
    underlying_issue: str = ""
    theme: str = ""
    sentiment: str = "unknown"
    priority: str = "medium"
    affected_capability: str = ""
    requested_need: str = ""
    recommended_action: str = ""
    failure_mode: str = ""
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FeedbackItem:
    feedback_id: str
    workspace: str
    channel: str
    timestamp: str
    text: str
    source_type: str = "live"
    persona: Optional[str] = None
    product_version: Optional[str] = None
    analysis: Optional[FeedbackAnalysis] = None
    lifecycle: str = LifecycleState.NOVEL.value
    source_id: Optional[str] = None
    wave_id: Optional[str] = None
    segment: Optional[str] = None
    rating: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def new(workspace: str, channel: str, text: str, source_type: str = "live") -> "FeedbackItem":
        return FeedbackItem(
            feedback_id=str(uuid.uuid4()),
            workspace=workspace,
            channel=channel,
            timestamp=datetime.utcnow().isoformat(timespec="seconds"),
            text=text,
            source_type=source_type,
        )

    def to_record(self) -> dict:
        rec = {
            "feedback_id": self.feedback_id,
            "workspace": self.workspace,
            "channel": self.channel,
            "timestamp": self.timestamp,
            "text": self.text,
            "source_type": self.source_type,
            "persona": self.persona,
            "product_version": self.product_version,
            "analysis": self.analysis.to_dict() if self.analysis else None,
            "lifecycle": self.lifecycle,
            "source_id": self.source_id,
            "wave_id": self.wave_id,
            "segment": self.segment,
            "rating": self.rating,
            "metadata": self.metadata,
        }
        return rec


@dataclass
class Memory:
    memory_id: str
    feedback_id: str
    workspace: str
    timestamp: str
    channel: str
    raw_text: str
    discovered_theme: str
    underlying_issue: str
    sentiment: str
    priority: str
    agent_interpretation: str
    affected_capability: str = ""
    failure_mode: str = ""
    lifecycle_state: str = LifecycleState.NOVEL.value
    product_change_id: Optional[str] = None
    expected_outcome: Optional[str] = None
    observed_outcome: Optional[str] = None
    recurrence_info: Optional[str] = None
    evidence_used: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ProductChange:
    change_id: str
    workspace: str
    title: str
    version: str
    date: str
    problem_issue: str
    expected_outcome: str
    observed_outcome: str = ""
    linked_theme: str = ""
    linked_capability: str = ""

    def to_dict(self) -> dict:
        return asdict(self)