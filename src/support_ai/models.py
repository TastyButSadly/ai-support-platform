from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class Action(StrEnum):
    DRAFT_FOR_OPERATOR = "DRAFT_FOR_OPERATOR"
    HUMAN_REVIEW = "HUMAN_REVIEW"
    ROUTE_ONLY = "ROUTE_ONLY"


@dataclass(frozen=True)
class Ticket:
    ticket_id: str
    channel: str
    text: str


@dataclass(frozen=True)
class TokenUsage:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True)
class Decision:
    topic: str | None
    confidence: float
    risk_level: str
    risk_reasons: list[str]
    route: str
    action: Action
    retrieved_document_id: str | None
    retrieval_score: float | None
    draft: str | None
    llm_called: bool
    fast_path_ms: float
    generation_ms: float | None
    token_usage: TokenUsage | None
    cost: float | None
    audit_id: str
    handoff_reason: str | None = None
    model: str | None = None
    trace: list[dict[str, Any]] = field(default_factory=list)
    audit_record: dict[str, Any] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["action"] = self.action.value
        return result
