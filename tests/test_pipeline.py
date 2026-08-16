from __future__ import annotations

import json

import pytest

from support_ai import pipeline
from support_ai.llm import DraftResult, OpenRouterDraftClient
from support_ai.models import Action, Ticket, TokenUsage


class FakeDraftClient:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.calls: list[dict[str, str]] = []
        self.error = error

    def generate(self, ticket_text: str, document_id: str, title: str, content: str) -> DraftResult:
        self.calls.append(
            {
                "ticket_text": ticket_text,
                "document_id": document_id,
                "title": title,
                "content": content,
            }
        )
        if self.error:
            raise self.error
        return DraftResult(
            text=f"Откройте настройки и выберите язык. Источник: {document_id}",
            model="fake-model",
            token_usage=TokenUsage(prompt_tokens=20, completion_tokens=10, total_tokens=30),
            cost=0.001,
        )


@pytest.fixture(autouse=True)
def isolated_audit(tmp_path, monkeypatch):
    audit_path = tmp_path / "audit.jsonl"
    monkeypatch.setattr(pipeline, "AUDIT_PATH", audit_path)
    return audit_path


def ticket(text: str) -> Ticket:
    return Ticket(ticket_id="test-ticket", channel="chat", text=text)


def test_safe_ticket_calls_llm_and_returns_sourced_draft(isolated_audit):
    client = FakeDraftClient()

    decision = pipeline.process_ticket(ticket("Как изменить язык приложения?"), client)

    assert decision.action is Action.DRAFT_FOR_OPERATOR
    assert decision.topic == "settings"
    assert decision.llm_called is True
    assert len(client.calls) == 1
    assert decision.retrieved_document_id == "KB-SET-001"
    assert "KB-SET-001" in decision.draft
    assert decision.cost == 0.001
    assert isolated_audit.exists()


def test_risky_ticket_never_calls_llm():
    client = FakeDraftClient()

    decision = pipeline.process_ticket(
        ticket("С меня дважды списали 12 000 рублей, верните деньги"), client
    )

    assert decision.action is Action.HUMAN_REVIEW
    assert decision.handoff_reason == "high_risk"
    assert decision.route == "payment_security_review"
    assert decision.llm_called is False
    assert client.calls == []


def test_possible_card_number_blocks_external_call():
    client = FakeDraftClient()

    decision = pipeline.process_ticket(
        ticket("Как изменить язык? Номер карты 4111 1111 1111 1111"), client
    )

    assert decision.action is Action.HUMAN_REVIEW
    assert "possible_payment_card" in decision.risk_reasons
    assert client.calls == []


def test_email_and_phone_are_redacted_before_llm_and_audit(isolated_audit):
    client = FakeDraftClient()
    original = "Как изменить язык приложения? Пишите ivan.petrov@example.com или +7 999 123-45-67"

    decision = pipeline.process_ticket(ticket(original), client)

    assert decision.action is Action.DRAFT_FOR_OPERATOR
    sent_text = client.calls[0]["ticket_text"]
    assert "ivan.petrov@example.com" not in sent_text
    assert "+7 999 123-45-67" not in sent_text
    assert "[EMAIL]" in sent_text
    assert "[PHONE]" in sent_text
    audit_text = isolated_audit.read_text(encoding="utf-8")
    assert "ivan.petrov@example.com" not in audit_text
    assert "+7 999 123-45-67" not in audit_text


def test_unknown_text_goes_to_human_review_without_llm():
    client = FakeDraftClient()

    decision = pipeline.process_ticket(ticket("ыва фыва непонятно"), client)

    assert decision.action is Action.HUMAN_REVIEW
    assert decision.handoff_reason == "low_classification_confidence"
    assert client.calls == []


def test_low_retrieval_score_does_not_start_generation(monkeypatch):
    client = FakeDraftClient()
    monkeypatch.setattr(pipeline, "RETRIEVAL_THRESHOLD", 1.01)

    decision = pipeline.process_ticket(ticket("Как изменить язык приложения?"), client)

    assert decision.action is Action.HUMAN_REVIEW
    assert decision.handoff_reason == "insufficient_knowledge"
    assert client.calls == []


def test_client_exception_becomes_llm_unavailable():
    client = FakeDraftClient(error=TimeoutError("provider timeout"))

    decision = pipeline.process_ticket(ticket("Как изменить язык приложения?"), client)

    assert decision.action is Action.HUMAN_REVIEW
    assert decision.handoff_reason == "llm_unavailable"
    assert decision.llm_called is True
    assert len(client.calls) == 1


def test_empty_llm_response_becomes_llm_unavailable():
    client = FakeDraftClient()

    def generate_empty(*_args):
        return DraftResult(text="", model="fake-model", token_usage=None, cost=None)

    client.generate = generate_empty

    decision = pipeline.process_ticket(ticket("Как изменить язык приложения?"), client)

    assert decision.action is Action.HUMAN_REVIEW
    assert decision.handoff_reason == "llm_unavailable"
    assert decision.llm_called is True


def test_audit_contains_versions_reasons_timings_and_no_secrets(isolated_audit):
    client = FakeDraftClient()

    pipeline.process_ticket(
        ticket("Верните деньги на карту 4111111111111111, почта secret@example.com"), client
    )

    record = json.loads(isolated_audit.read_text(encoding="utf-8"))
    assert record["versions"]["pipeline"]
    assert record["risk_reasons"]
    assert record["fast_path_ms"] >= 0
    assert record["cleaned_text"]
    serialized = json.dumps(record, ensure_ascii=False)
    assert "secret@example.com" not in serialized
    assert "4111111111111111" not in serialized
    assert "SECRET_SENTINEL" not in serialized


def test_missing_environment_fails_fast(monkeypatch):
    for name in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL"):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(RuntimeError, match="обязательные"):
        OpenRouterDraftClient.from_env()
