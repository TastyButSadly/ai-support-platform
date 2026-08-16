from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv

from support_ai.llm import OpenRouterDraftClient
from support_ai.models import Action, Ticket
from support_ai.pipeline import process_ticket

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_LIVE_LLM") != "1",
    reason="Для платного smoke-теста задайте RUN_LIVE_LLM=1",
)


def test_live_openrouter_generates_draft(tmp_path, monkeypatch):
    load_dotenv()
    from support_ai import pipeline

    monkeypatch.setattr(pipeline, "AUDIT_PATH", tmp_path / "audit.jsonl")
    client = OpenRouterDraftClient.from_env()
    decision = process_ticket(
        Ticket(ticket_id="live-smoke", channel="chat", text="Как изменить язык приложения?"),
        client,
    )

    assert decision.action is Action.DRAFT_FOR_OPERATOR
    assert decision.draft
    assert decision.llm_called is True
    assert decision.token_usage is not None
