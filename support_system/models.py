from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Priority = Literal["low", "moderate", "high"]
Sentiment = Literal["positive", "neutral", "frustrated", "angry"]


class Classification(BaseModel):
    category: str = Field(description="Short product-support category")
    urgency: Priority
    sentiment: Sentiment
    priority: Priority
    summary: str
    is_product_related: bool
    rationale: str


class Resolution(BaseModel):
    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    source_filenames: list[str] = Field(default_factory=list)
    grounded: bool


class WorkflowResult(BaseModel):
    ticket_id: str
    user_id: str
    status: Literal["resolved", "escalated"]
    classification: Classification
    resolution: Resolution
    customer_message: str
    escalation_reason: str = ""
    email_status: str = "not_required"
    passages: list[dict] = Field(default_factory=list)
    stages: list[dict] = Field(default_factory=list)
    timings_ms: dict[str, float] = Field(default_factory=dict)
