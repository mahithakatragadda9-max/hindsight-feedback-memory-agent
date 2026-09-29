from dataclasses import dataclass
from models import Memory, FeedbackAnalysis


STRONG_RELEVANCE_THRESHOLD = 0.75


@dataclass
class Evidence:
    memory: Memory
    relevance: float
    same_issue: bool
    reason: str
    is_current_session: bool = False

    @property
    def is_strong(self) -> bool:
        return self.same_issue and self.relevance >= STRONG_RELEVANCE_THRESHOLD

    def to_dict(self) -> dict:
        return {
            "memory_id": self.memory.memory_id,
            "feedback_id": self.memory.feedback_id,
            "raw_text": self.memory.raw_text,
            "timestamp": self.memory.timestamp,
            "channel": self.memory.channel,
            "theme": self.memory.discovered_theme,
            "underlying_issue": self.memory.underlying_issue,
            "lifecycle_state": self.memory.lifecycle_state,
            "relevance": self.relevance,
            "same_issue": self.same_issue,
            "reason": self.reason,
            "is_current_session": self.is_current_session,
        }


def build_evidence(
    analysis: FeedbackAnalysis,
    memories: list[Memory],
    evaluations: list[dict],
    current_feedback_id: str,
) -> list[Evidence]:
    eval_map = {e.get("memory_id"): e for e in evaluations if e.get("memory_id")}
    evidence: list[Evidence] = []
    for m in memories:
        e = eval_map.get(m.memory_id, {})
        try:
            relevance = float(e.get("relevance", 0.0))
        except (TypeError, ValueError):
            relevance = 0.0
        evidence.append(
            Evidence(
                memory=m,
                relevance=relevance,
                same_issue=bool(e.get("same_issue", False)),
                reason=e.get("reason", "Not evaluated."),
                is_current_session=(m.feedback_id == current_feedback_id),
            )
        )
    evidence.sort(key=lambda x: x.relevance, reverse=True)
    return evidence
