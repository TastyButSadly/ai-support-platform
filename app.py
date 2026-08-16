from __future__ import annotations

import html
import json
import os
import uuid
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from support_ai.llm import OpenRouterDraftClient
from support_ai.models import Action, Ticket
from support_ai.pipeline import process_ticket

ROOT = Path(__file__).resolve().parent

ACTION_LABELS = {
    Action.DRAFT_FOR_OPERATOR: "Черновик готов",
    Action.HUMAN_REVIEW: "Ручная проверка",
    Action.ROUTE_ONLY: "Только маршрут",
}

RISK_LABELS = {"low": "Низкий", "high": "Высокий"}

TOPIC_LABELS = {
    "settings": "Настройки",
    "account_access": "Доступ к аккаунту",
    "billing": "Подписка и платежи",
    "service_incident": "Сбой сервиса",
}

ROUTE_LABELS = {
    "general_support": "Общая поддержка",
    "account_support": "Доступ к аккаунту",
    "billing_support": "Платежная поддержка",
    "incident_support": "Инциденты",
    "payment_security_review": "Платежная безопасность",
    "account_security_review": "Безопасность аккаунта",
    "specialist_review": "Проверка специалистом",
}

HANDOFF_LABELS = {
    "high_risk": "Высокий риск",
    "low_classification_confidence": "Низкая уверенность классификации",
    "insufficient_knowledge": "Нет подходящей статьи",
    "llm_unavailable": "Сервис генерации недоступен",
}

RISK_REASON_LABELS = {
    "disputed_charge": "Спорное списание",
    "refund": "Возврат",
    "fraud": "Мошенничество",
    "account_takeover": "Захват аккаунта",
    "legal_claim": "Претензия",
    "security_threat": "Угроза безопасности",
    "possible_payment_card": "Возможный номер карты",
}


def _setting(name: str) -> str:
    value = os.getenv(name, "").strip()
    if value:
        return value
    try:
        return str(st.secrets.get(name, "")).strip()
    except FileNotFoundError:
        return ""


@st.cache_resource
def llm_client() -> OpenRouterDraftClient:
    load_dotenv(ROOT / ".env")
    return OpenRouterDraftClient(
        api_key=_setting("LLM_API_KEY"),
        base_url=_setting("LLM_BASE_URL"),
        model=_setting("LLM_MODEL"),
    )


@st.cache_data
def load_json(name: str) -> list[dict[str, str]]:
    with (ROOT / "data" / name).open(encoding="utf-8") as file:
        return json.load(file)


def render_route(decision) -> None:
    rows = [
        ("Тема", TOPIC_LABELS.get(decision.topic, "Не определена")),
        ("Уверенность", f"{decision.confidence:.1%}"),
        ("Риск", RISK_LABELS.get(decision.risk_level, decision.risk_level)),
        ("Очередь", ROUTE_LABELS.get(decision.route, decision.route)),
        ("Действие", ACTION_LABELS[decision.action]),
        ("Вызов LLM", "Да" if decision.llm_called else "Нет"),
    ]
    markup = "".join(
        f'<div class="detail-row"><span>{html.escape(label)}</span>'
        f"<strong>{html.escape(value)}</strong></div>"
        for label, value in rows
    )
    st.markdown(f'<div class="details">{markup}</div>', unsafe_allow_html=True)


st.set_page_config(page_title="Обработка тикета", layout="wide")
st.markdown(
    """
    <style>
        .stApp * { border-radius: 0 !important; }
        .stApp { background: #f7f8fa; color: #1f2937; }
        header[data-testid="stHeader"] { background: transparent; }
        .block-container { max-width: 1440px; padding: 1.5rem 2rem 2rem; }
        h1 {
            font-size: 1.25rem !important;
            font-weight: 650 !important;
            margin: 0 0 1rem !important;
        }
        h2, h3 { font-size: 1rem !important; font-weight: 650 !important; }
        [data-testid="stVerticalBlockBorderWrapper"] {
            background: #ffffff;
            border: 1px solid #e2e5e9 !important;
            box-shadow: none;
        }
        [data-testid="stWidgetLabel"] p { font-size: 0.78rem; color: #5f6773; }
        .stTextArea textarea, [data-baseweb="select"] > div {
            background: #ffffff;
            border-color: #cfd4da;
        }
        .stButton button {
            font-weight: 600;
            min-height: 2.4rem;
        }
        .status-line {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            min-height: 2.4rem;
        }
        .status-line strong { font-size: 1rem; font-weight: 650; }
        .status {
            display: inline-block;
            padding: 0.22rem 0.55rem;
            border: 1px solid #cfd4da;
            color: #344054;
            font-size: 0.76rem;
            font-weight: 600;
            background: #f8fafc;
        }
        .status.review { color: #8a4b08; border-color: #e7c995; background: #fff8eb; }
        .status.ready { color: #17633a; border-color: #a8d5bb; background: #effaf3; }
        .details { border-top: 1px solid #eceef1; }
        .detail-row {
            display: grid;
            grid-template-columns: minmax(120px, 0.8fr) 1.4fr;
            gap: 1rem;
            padding: 0.65rem 0;
            border-bottom: 1px solid #eceef1;
            font-size: 0.84rem;
        }
        .detail-row span { color: #69717d; }
        .detail-row strong { font-weight: 550; overflow-wrap: anywhere; }
        [data-baseweb="tab-list"] { gap: 1.25rem; border-bottom: 1px solid #e2e5e9; }
        [data-baseweb="tab"] { padding-left: 0; padding-right: 0; }
        footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

try:
    client = llm_client()
except RuntimeError as error:
    st.error(str(error))
    st.stop()

presets = load_json("demo_tickets.json")
documents = {item["id"]: item for item in load_json("knowledge_base.json")}
scenario_names = [item["name"] for item in presets] + ["Собственное обращение"]

st.title("Обработка тикета")
ticket_panel, result_panel = st.columns([0.9, 1.25], gap="medium")

with ticket_panel:
    with st.container(border=True):
        scenario = st.selectbox("Сценарий", scenario_names)
        selected = next((item for item in presets if item["name"] == scenario), None)
        channel_options = ["chat", "email", "web", "mobile"]
        default_channel = selected["channel"] if selected else "chat"
        channel = st.selectbox(
            "Канал",
            channel_options,
            index=channel_options.index(default_channel),
        )
        text = st.text_area(
            "Обращение",
            value=selected["text"] if selected else "",
            height=220,
            placeholder="Текст обращения",
        )
        submitted = st.button(
            "Обработать",
            type="primary",
            width="stretch",
            disabled=not text.strip(),
        )

decision = None
if submitted:
    ticket = Ticket(ticket_id=str(uuid.uuid4()), channel=channel, text=text.strip())
    with st.spinner("Обработка"):
        decision = process_ticket(ticket, client)

if decision:
    with result_panel:
        with st.container(border=True):
            status_class = "ready" if decision.action is Action.DRAFT_FOR_OPERATOR else "review"
            st.markdown(
                '<div class="status-line">'
                f"<strong>Результат</strong>"
                f'<span class="status {status_class}">{ACTION_LABELS[decision.action]}</span>'
                "</div>",
                unsafe_allow_html=True,
            )

            response_tab, route_tab, audit_tab = st.tabs(["Ответ", "Маршрут", "Аудит"])

            with response_tab:
                if decision.draft:
                    st.text_area(
                        "Черновик ответа",
                        value=decision.draft,
                        height=180,
                        key=f"draft-{decision.audit_id}",
                    )
                else:
                    handoff = HANDOFF_LABELS.get(
                        decision.handoff_reason,
                        decision.handoff_reason,
                    )
                    st.write(f"Причина передачи: {handoff}")
                    if decision.risk_reasons:
                        reasons = [
                            RISK_REASON_LABELS.get(reason, reason)
                            for reason in decision.risk_reasons
                        ]
                        st.write("Причины риска: " + ", ".join(reasons))

                if decision.retrieved_document_id:
                    document = documents[decision.retrieved_document_id]
                    score = decision.retrieval_score or 0.0
                    st.markdown(
                        f"**Источник:** {document['title']} "
                        f"(`{decision.retrieved_document_id}`, {score:.0%})"
                    )

            with route_tab:
                render_route(decision)

            with audit_tab:
                timing_left, timing_right = st.columns(2)
                timing_left.metric("Быстрый путь", f"{decision.fast_path_ms:.1f} мс")
                timing_right.metric(
                    "Генерация",
                    (
                        f"{decision.generation_ms:.1f} мс"
                        if decision.generation_ms is not None
                        else "—"
                    ),
                )
                if decision.token_usage:
                    st.write(
                        f"Токены: {decision.token_usage.prompt_tokens} / "
                        f"{decision.token_usage.completion_tokens} / "
                        f"{decision.token_usage.total_tokens}"
                    )
                if decision.cost is not None:
                    st.write(f"Стоимость: {decision.cost}")
                st.dataframe(decision.trace, width="stretch", hide_index=True)
                with st.expander("JSON"):
                    st.json(decision.audit_record)
