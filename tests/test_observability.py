from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest

from support_system.models import Classification, Resolution
from support_system.rag import LocalFaissStore
from support_system.sheets_mirror import TicketOutbox
from support_system.storage import TicketLogger
from support_system.workflow import SupportWorkflow


EXPECTED_STAGES = {
    "product_retrieval",
    "classification",
    "support_retrieval_synthesis",
    "resolution",
    "local_persistence",
    "escalation_email_handling",
}


def classification() -> Classification:
    return Classification(
        category="refund",
        urgency="moderate",
        sentiment="neutral",
        priority="moderate",
        summary="Refund request",
        is_product_related=True,
        rationale="Matched policy",
    )


def resolution() -> Resolution:
    return Resolution(
        answer="Refunds take five business days.",
        confidence=0.9,
        source_filenames=["refund.pdf"],
        grounded=True,
    )


def append_ticket(logger: TicketLogger, ticket_id: str = "T-OBS") -> None:
    logger.append(
        ticket_id=ticket_id,
        user_id="customer",
        query="Where is my refund?",
        classification=classification(),
        resolution=resolution(),
        status="resolved",
        escalation_reason="",
    )


def test_update_observability_persists_valid_timings_after_ticket_and_emits_snapshot(
    tmp_path: Path,
):
    snapshots: list[dict[str, str]] = []
    logger = TicketLogger(tmp_path / "tickets.csv", event_sink=snapshots.append)
    append_ticket(logger)
    snapshots.clear()
    stage_latencies = {
        "product_retrieval": 1.25,
        "classification": 2.5,
        "support_retrieval_synthesis": 3.75,
        "resolution": 4.0,
        "local_persistence": 5.5,
        "escalation_email_handling": 0.0,
    }

    assert logger.update_observability("T-OBS", stage_latencies, total_latency_ms=17.0)

    row = logger.read_all()[0]
    assert json.loads(row["stage_latencies_ms"]) == stage_latencies
    assert float(row["total_latency_ms"]) == 17.0
    assert snapshots == [row]


def test_update_observability_is_safe_for_missing_tickets_and_invalid_values(tmp_path: Path):
    snapshots: list[dict[str, str]] = []
    logger = TicketLogger(tmp_path / "tickets.csv", event_sink=snapshots.append)

    assert logger.update_observability("missing", {"classification": 1.0}, 1.0) is False
    assert snapshots == []

    append_ticket(logger)
    for invalid in (-1.0, math.inf, math.nan):
        with pytest.raises(ValueError, match="finite and non-negative"):
            logger.update_observability("T-OBS", {"classification": invalid}, 1.0)
    with pytest.raises(ValueError, match="finite and non-negative"):
        logger.update_observability("T-OBS", {"classification": 1.0}, math.inf)


def test_observability_snapshot_is_distinct_from_google_sheets_sync_state(tmp_path: Path):
    outbox = TicketOutbox(tmp_path / "outbox.jsonl")
    logger = TicketLogger(tmp_path / "tickets.csv", event_sink=outbox.enqueue)
    append_ticket(logger)

    assert logger.update_observability("T-OBS", {"classification": 2.0}, 3.0)

    records = outbox.read_all()
    assert len(records) == 2
    latest = records[-1]
    assert latest["payload"]["stage_latencies_ms"] == '{"classification": 2.0}'
    assert latest["payload"]["total_latency_ms"] == "3.0"
    assert latest["state"] == "pending"
    assert "state" not in logger.read_all()[0]


def test_observability_schema_migration_preserves_distinct_status_fields(tmp_path: Path):
    path = tmp_path / "tickets.csv"
    legacy_fields = [
        "ticket_id",
        "status",
        "escalation_delivery_state",
        "review_email_status",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=legacy_fields)
        writer.writeheader()
        writer.writerow(
            {
                "ticket_id": "T-OLD",
                "status": "escalated",
                "escalation_delivery_state": "sent",
                "review_email_status": "failed:mailbox",
            }
        )

    row = TicketLogger(path).read_all()[0]

    assert row["status"] == "escalated"
    assert row["escalation_delivery_state"] == "sent"
    assert row["review_email_status"] == "failed:mailbox"
    assert row["stage_latencies_ms"] == ""
    assert row["total_latency_ms"] == ""


class StepClock:
    def __init__(self, step: float = 0.001):
        self.value = -step
        self.step = step

    def __call__(self) -> float:
        self.value += self.step
        return self.value


class QueryAgent:
    async def classify(self, query, product_hits):
        return classification()


class KnowledgeAgent:
    async def retrieve(self, query, classified, store):
        return store.search(query, top_k=3)


class ResolutionAgent:
    async def resolve(self, query, classified, passages):
        return Resolution(
            answer="Refunds take five business days.",
            confidence=0.9,
            source_filenames=[item["source"] for item in passages],
            grounded=bool(passages),
        )


class Mailer:
    def send(self, **payload):
        raise AssertionError("resolved ticket must not send email")


async def test_workflow_returns_and_persists_complete_stage_and_total_timings(tmp_path: Path):
    product_store = LocalFaissStore(tmp_path / "product")
    support_store = LocalFaissStore(tmp_path / "support")
    for store in (product_store, support_store):
        store.add_texts(
            ["Eligible refunds take five business days."],
            [{"source": "refund.pdf", "kind": "policy"}],
        )
    logger = TicketLogger(tmp_path / "tickets.csv")
    workflow = SupportWorkflow(
        query_agent=QueryAgent(),
        knowledge_agent=KnowledgeAgent(),
        resolution_agent=ResolutionAgent(),
        product_store=product_store,
        support_store=support_store,
        ticket_logger=logger,
        escalation_mailer=Mailer(),
        perf_counter=StepClock(),
    )

    result = await workflow.process("When will my eligible refund arrive?", "customer")

    assert set(result.timings_ms) == EXPECTED_STAGES | {"total_latency_ms"}
    assert all(math.isfinite(value) and value >= 0 for value in result.timings_ms.values())
    assert result.timings_ms["total_latency_ms"] >= sum(
        result.timings_ms[name] for name in EXPECTED_STAGES
    )
    row = logger.read_all()[0]
    assert json.loads(row["stage_latencies_ms"]) == {
        name: result.timings_ms[name] for name in EXPECTED_STAGES
    }
    assert float(row["total_latency_ms"]) == result.timings_ms["total_latency_ms"]


class FailingQueryAgent:
    async def classify(self, query, product_hits):
        raise RuntimeError("classification unavailable")


async def test_workflow_exception_does_not_persist_fabricated_completion_timings(tmp_path: Path):
    logger = TicketLogger(tmp_path / "tickets.csv")
    workflow = SupportWorkflow(
        query_agent=FailingQueryAgent(),
        knowledge_agent=KnowledgeAgent(),
        resolution_agent=ResolutionAgent(),
        product_store=LocalFaissStore(tmp_path / "product"),
        support_store=LocalFaissStore(tmp_path / "support"),
        ticket_logger=logger,
        escalation_mailer=Mailer(),
        perf_counter=StepClock(),
    )

    with pytest.raises(RuntimeError, match="classification unavailable"):
        await workflow.process("Where is my refund?", "customer")

    assert logger.read_all() == []
