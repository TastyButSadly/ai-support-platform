"""Прототип обработки обращений поддержки."""

from support_ai.models import Action, Decision, Ticket
from support_ai.pipeline import process_ticket

__all__ = ["Action", "Decision", "Ticket", "process_ticket"]
