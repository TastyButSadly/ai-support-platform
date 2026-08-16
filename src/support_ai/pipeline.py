from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

from support_ai.llm import DraftClient
from support_ai.models import Action, Decision, Ticket

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
AUDIT_PATH = PROJECT_ROOT / ".local" / "audit.jsonl"
CONFIDENCE_THRESHOLD = 0.55
MARGIN_THRESHOLD = 0.15
RETRIEVAL_THRESHOLD = 0.20
PIPELINE_VERSION = "0.1.0"
CLASSIFIER_VERSION = "tfidf-logreg-synthetic-v1"
RETRIEVER_VERSION = "tfidf-kb-v1"

EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
PHONE_RE = re.compile(r"(?<!\d)(?:\+7|8)[\s()\-]*\d{3}[\s()\-]*\d{3}[\s\-]*\d{2}[\s\-]*\d{2}(?!\d)")
CARD_CANDIDATE_RE = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")

RISK_PATTERNS: dict[str, tuple[str, ...]] = {
    "disputed_charge": (
        "дважды списал",
        "двойное списание",
        "спорное списание",
        "не совершал платеж",
    ),
    "refund": ("верните деньги", "требую возврат", "оформить возврат", "возврат денег"),
    "fraud": ("мошен", "украли деньги", "чужая операция"),
    "account_takeover": ("взломали", "захват аккаунта", "чужой вошел", "потерял контроль"),
    "legal_claim": ("претензия", "подам в суд", "роспотребнадзор", "адвокат"),
    "security_threat": ("угроза безопасности", "утечка данных", "уязвимость", "вымогател"),
}

ROUTES = {
    "settings": "general_support",
    "account_access": "account_support",
    "billing": "billing_support",
    "service_incident": "incident_support",
}


class _PipelineAssets:
    def __init__(self) -> None:
        training = _load_json(DATA_DIR / "training_tickets.json")
        self.documents: list[dict[str, str]] = _load_json(DATA_DIR / "knowledge_base.json")

        self.classifier_vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), lowercase=True
        )
        train_matrix = self.classifier_vectorizer.fit_transform(
            [item["text"] for item in training]
        )
        self.classifier = LogisticRegression(class_weight="balanced", max_iter=1000, C=2.0)
        self.classifier.fit(train_matrix, [item["topic"] for item in training])

        self.retrieval_vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), lowercase=True
        )
        searchable = [f"{doc['title']} {doc['content']}" for doc in self.documents]
        self.document_matrix = self.retrieval_vectorizer.fit_transform(searchable)

    def classify(self, text: str) -> tuple[str, float, float]:
        probabilities = self.classifier.predict_proba(
            self.classifier_vectorizer.transform([text])
        )[0]
        ranked = sorted(
            zip(self.classifier.classes_, probabilities, strict=True),
            key=lambda item: item[1],
            reverse=True,
        )
        return ranked[0][0], float(ranked[0][1]), float(ranked[0][1] - ranked[1][1])

    def retrieve(self, text: str, topic: str) -> tuple[dict[str, str], float]:
        query = self.retrieval_vectorizer.transform([text])
        scores = cosine_similarity(query, self.document_matrix)[0]
        eligible = [
            (index, float(score))
            for index, score in enumerate(scores)
            if self.documents[index]["topic"] == topic
        ]
        index, score = max(eligible, key=lambda item: item[1])
        return self.documents[index], score


_ASSETS: _PipelineAssets | None = None


def _assets() -> _PipelineAssets:
    global _ASSETS
    if _ASSETS is None:
        _ASSETS = _PipelineAssets()
    return _ASSETS


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _risk_reasons(text: str) -> list[str]:
    normalized = text.lower().replace("ё", "е")
    reasons = [
        reason
        for reason, patterns in RISK_PATTERNS.items()
        if any(pattern in normalized for pattern in patterns)
    ]
    for candidate in CARD_CANDIDATE_RE.findall(text):
        digits = re.sub(r"\D", "", candidate)
        if 13 <= len(digits) <= 19:
            reasons.append("possible_payment_card")
            break
    return reasons


def _redact(text: str) -> str:
    without_cards = CARD_CANDIDATE_RE.sub("[POSSIBLE_CARD]", text)
    return PHONE_RE.sub("[PHONE]", EMAIL_RE.sub("[EMAIL]", without_cards))


def _route_for_risk(reasons: list[str]) -> str:
    payment_reasons = {"disputed_charge", "refund", "fraud", "possible_payment_card"}
    if payment_reasons.intersection(reasons):
        return "payment_security_review"
    if "account_takeover" in reasons:
        return "account_security_review"
    return "specialist_review"


def _write_audit(record: dict[str, Any]) -> None:
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with AUDIT_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def _finalize(
    *,
    ticket: Ticket,
    cleaned_text: str,
    started: float,
    audit_id: str,
    topic: str | None,
    confidence: float,
    risk_level: str,
    risk_reasons: list[str],
    route: str,
    action: Action,
    trace: list[dict[str, Any]],
    handoff_reason: str | None = None,
    document: dict[str, str] | None = None,
    retrieval_score: float | None = None,
    draft: str | None = None,
    llm_called: bool = False,
    generation_ms: float | None = None,
    token_usage: Any = None,
    cost: float | None = None,
    model: str | None = None,
    fast_path_ms: float | None = None,
) -> Decision:
    total_ms = (time.perf_counter() - started) * 1000
    fast_ms = total_ms if fast_path_ms is None else fast_path_ms
    record = {
        "audit_id": audit_id,
        "created_at": datetime.now(UTC).isoformat(),
        "ticket_id": ticket.ticket_id,
        "channel": ticket.channel,
        "cleaned_text": cleaned_text,
        "topic": topic,
        "confidence": round(confidence, 6),
        "risk_level": risk_level,
        "risk_reasons": risk_reasons,
        "route": route,
        "action": action.value,
        "handoff_reason": handoff_reason,
        "retrieved_document_id": document["id"] if document else None,
        "retrieval_score": round(retrieval_score, 6) if retrieval_score is not None else None,
        "draft": draft,
        "llm_called": llm_called,
        "fast_path_ms": round(fast_ms, 3),
        "generation_ms": round(generation_ms, 3) if generation_ms is not None else None,
        "token_usage": asdict(token_usage) if token_usage else None,
        "cost": cost,
        "model": model,
        "versions": {
            "pipeline": PIPELINE_VERSION,
            "classifier": CLASSIFIER_VERSION,
            "retriever": RETRIEVER_VERSION,
        },
        "trace": trace,
    }
    _write_audit(record)
    return Decision(
        topic=topic,
        confidence=confidence,
        risk_level=risk_level,
        risk_reasons=risk_reasons,
        route=route,
        action=action,
        retrieved_document_id=document["id"] if document else None,
        retrieval_score=retrieval_score,
        draft=draft,
        llm_called=llm_called,
        fast_path_ms=fast_ms,
        generation_ms=generation_ms,
        token_usage=token_usage,
        cost=cost,
        audit_id=audit_id,
        handoff_reason=handoff_reason,
        model=model,
        trace=trace,
        audit_record=record,
    )


def process_ticket(ticket: Ticket, llm_client: DraftClient) -> Decision:
    started = time.perf_counter()
    audit_id = str(uuid.uuid4())
    trace: list[dict[str, Any]] = []

    step_started = time.perf_counter()
    reasons = _risk_reasons(ticket.text)
    cleaned_text = _redact(ticket.text)
    trace.append(
        {
            "step": "risk_and_pii",
            "ms": round((time.perf_counter() - step_started) * 1000, 3),
        }
    )
    if reasons:
        return _finalize(
            ticket=ticket,
            cleaned_text=cleaned_text,
            started=started,
            audit_id=audit_id,
            topic=None,
            confidence=0.0,
            risk_level="high",
            risk_reasons=reasons,
            route=_route_for_risk(reasons),
            action=Action.HUMAN_REVIEW,
            handoff_reason="high_risk",
            trace=trace,
        )

    step_started = time.perf_counter()
    topic, confidence, margin = _assets().classify(cleaned_text)
    trace.append(
        {
            "step": "classification",
            "ms": round((time.perf_counter() - step_started) * 1000, 3),
            "margin": round(margin, 6),
        }
    )
    route = ROUTES[topic]
    if confidence < CONFIDENCE_THRESHOLD or margin < MARGIN_THRESHOLD:
        return _finalize(
            ticket=ticket,
            cleaned_text=cleaned_text,
            started=started,
            audit_id=audit_id,
            topic=topic,
            confidence=confidence,
            risk_level="low",
            risk_reasons=[],
            route=route,
            action=Action.HUMAN_REVIEW,
            handoff_reason="low_classification_confidence",
            trace=trace,
        )

    step_started = time.perf_counter()
    document, retrieval_score = _assets().retrieve(cleaned_text, topic)
    trace.append(
        {
            "step": "knowledge_retrieval",
            "ms": round((time.perf_counter() - step_started) * 1000, 3),
        }
    )
    if retrieval_score < RETRIEVAL_THRESHOLD:
        return _finalize(
            ticket=ticket,
            cleaned_text=cleaned_text,
            started=started,
            audit_id=audit_id,
            topic=topic,
            confidence=confidence,
            risk_level="low",
            risk_reasons=[],
            route=route,
            action=Action.HUMAN_REVIEW,
            handoff_reason="insufficient_knowledge",
            document=document,
            retrieval_score=retrieval_score,
            trace=trace,
        )

    fast_path_ms = (time.perf_counter() - started) * 1000
    generation_started = time.perf_counter()
    try:
        generated = llm_client.generate(
            cleaned_text,
            document["id"],
            document["title"],
            document["content"],
        )
        generation_ms = (time.perf_counter() - generation_started) * 1000
        draft = generated.text.strip()
        if not draft:
            raise RuntimeError("LLM returned an empty draft")
        if document["id"] not in draft:
            draft = f"{draft}\n\nИсточник: {document['title']} ({document['id']})"
        trace.append({"step": "llm_generation", "ms": round(generation_ms, 3)})
        return _finalize(
            ticket=ticket,
            cleaned_text=cleaned_text,
            started=started,
            audit_id=audit_id,
            topic=topic,
            confidence=confidence,
            risk_level="low",
            risk_reasons=[],
            route=route,
            action=Action.DRAFT_FOR_OPERATOR,
            document=document,
            retrieval_score=retrieval_score,
            draft=draft,
            llm_called=True,
            generation_ms=generation_ms,
            token_usage=generated.token_usage,
            cost=generated.cost,
            model=generated.model,
            fast_path_ms=fast_path_ms,
            trace=trace,
        )
    except Exception:  # Внешний клиент не должен прерывать обработку обращения.
        generation_ms = (time.perf_counter() - generation_started) * 1000
        trace.append({"step": "llm_generation", "ms": round(generation_ms, 3), "status": "failed"})
        return _finalize(
            ticket=ticket,
            cleaned_text=cleaned_text,
            started=started,
            audit_id=audit_id,
            topic=topic,
            confidence=confidence,
            risk_level="low",
            risk_reasons=[],
            route=route,
            action=Action.HUMAN_REVIEW,
            handoff_reason="llm_unavailable",
            document=document,
            retrieval_score=retrieval_score,
            llm_called=True,
            generation_ms=generation_ms,
            fast_path_ms=fast_path_ms,
            trace=trace,
        )
