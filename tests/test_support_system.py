from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import csv
from pathlib import Path

import faiss
import pytest

from support_system.email_service import EscalationMailer
from support_system.ingestion import MAX_PDF_BYTES, ingest_pdf
from support_system.models import Classification, Resolution
from support_system.rag import LocalFaissStore
from support_system.storage import TicketLogger
from support_system.workflow import SupportWorkflow


class FakeQueryAgent:
    def __init__(self, priority: str = "moderate", related: bool = True) -> None:
        self.priority = priority
        self.related = related

    async def classify(self, query: str, product_context: list[dict]) -> Classification:
        return Classification(
            category="refund",
            urgency=self.priority,
            sentiment="frustrated",
            priority=self.priority,
            summary=query,
            is_product_related=self.related,
            rationale="Matched refund policy.",
        )


class FakeKnowledgeAgent:
    async def retrieve(self, query: str, classification: Classification, store: LocalFaissStore):
        return store.search(query, top_k=3)

    async def synthesize(
        self, query: str, classification: Classification, passages: list[dict]
    ) -> list[dict]:
        return passages


class FakeRetrievalClient:
    def __init__(self, product_hits: list[dict], support_hits: list[dict]) -> None:
        self.product_hits = product_hits
        self.support_hits = support_hits
        self.calls: list[str] = []

    async def search_product_scope(self, query: str, top_k: int = 4) -> list[dict]:
        self.calls.append("search_product_scope")
        return self.product_hits

    async def search_support_knowledge(self, query: str, top_k: int = 6) -> list[dict]:
        self.calls.append("search_support_knowledge")
        return self.support_hits


class FakeResolutionAgent:
    def __init__(self, confidence: float) -> None:
        self.confidence = confidence

    async def resolve(self, query: str, classification: Classification, passages: list[dict]) -> Resolution:
        return Resolution(
            answer="Your eligible refund will be processed within five business days.",
            confidence=self.confidence,
            source_filenames=[item["source"] for item in passages],
            grounded=bool(passages),
        )


class FabricatedCitationAgent(FakeResolutionAgent):
    async def resolve(self, query: str, classification: Classification, passages: list[dict]) -> Resolution:
        return Resolution(
            answer="Unsupported answer.",
            confidence=0.99,
            source_filenames=["invented-policy.pdf"],
            grounded=True,
        )


class FakeEscalationMailer:
    def __init__(self) -> None:
        self.calls = []

    def send(self, **payload):
        self.calls.append(payload)
        return {"sent": True, "detail": "test"}


class FailingLogger:
    def append(self, **payload):
        raise OSError("disk full")


def _build_store(path: Path) -> LocalFaissStore:
    store = LocalFaissStore(path)
    store.add_texts(
        ["Refunds for eligible purchases are processed within five business days."],
        [{"source": "refund_policy.pdf", "page": 2, "kind": "policy"}],
    )
    return store


def test_faiss_store_persists_source_and_returns_relevant_passage(tmp_path: Path):
    store = _build_store(tmp_path / "kb")
    reloaded = LocalFaissStore(tmp_path / "kb")

    hits = reloaded.search("How long will an eligible refund take?", top_k=1)

    assert hits
    assert hits[0]["source"] == "refund_policy.pdf"
    assert hits[0]["page"] == 2
    assert "five business days" in hits[0]["text"]


def test_faiss_store_recovers_from_index_metadata_mismatch(tmp_path: Path):
    directory = tmp_path / "recoverable-kb"
    store = _build_store(directory)
    faiss.write_index(faiss.IndexFlatIP(store.embedder.dimension), str(store.index_path))

    recovered = LocalFaissStore(directory)

    assert recovered.index.ntotal == len(recovered.records) == 1
    assert recovered.search("refund five business days", top_k=1)


def test_faiss_store_serializes_concurrent_writes(tmp_path: Path):
    directory = tmp_path / "shared-kb"
    first = LocalFaissStore(directory)
    second = LocalFaissStore(directory)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(first.add_texts, ["refund policy alpha"], [{"source": "a.pdf", "kind": "policy"}]),
            pool.submit(second.add_texts, ["login guide beta"], [{"source": "b.pdf", "kind": "faq"}]),
        ]
        assert [future.result() for future in futures] == [1, 1]

    reloaded = LocalFaissStore(directory)
    assert reloaded.index.ntotal == 2
    assert {record["source"] for record in reloaded.records} == {"a.pdf", "b.pdf"}


def test_ticket_logger_writes_required_audit_fields(tmp_path: Path):
    logger = TicketLogger(tmp_path / "tickets.csv")
    logger.append(
        ticket_id="T-1",
        user_id="customer-7",
        query="Where is my refund?",
        classification=Classification(
            category="refund",
            urgency="moderate",
            sentiment="frustrated",
            priority="moderate",
            summary="Refund status request",
            is_product_related=True,
            rationale="Matched policy",
        ),
        resolution=Resolution(
            answer="It takes five business days.",
            confidence=0.91,
            source_filenames=["refund_policy.pdf"],
            grounded=True,
        ),
        status="resolved",
        escalation_reason="",
    )

    rows = logger.read_all()
    assert rows[0]["user_id"] == "customer-7"
    assert rows[0]["priority"] == "moderate"
    assert rows[0]["status"] == "resolved"
    assert rows[0]["timestamp_utc"].endswith("Z")


def test_ticket_logger_neutralizes_spreadsheet_formulas(tmp_path: Path):
    logger = TicketLogger(tmp_path / "tickets.csv")
    classification = Classification(
        category="refund",
        urgency="low",
        sentiment="neutral",
        priority="low",
        summary="@SUM(1+1)",
        is_product_related=True,
        rationale="Matched policy",
    )
    logger.append(
        ticket_id="T-CSV",
        user_id="+cmd|' /C calc'!A0",
        query='=HYPERLINK("https://example.invalid")',
        classification=classification,
        resolution=Resolution(
            answer="-malicious-formula",
            confidence=0.9,
            source_filenames=["refund.pdf"],
            grounded=True,
        ),
        status="resolved",
        escalation_reason="",
    )

    row = logger.read_all()[0]
    assert row["user_id"].startswith("'+")
    assert row["query"].startswith("'=")
    assert row["summary"].startswith("'@")
    assert row["answer"].startswith("'-")


async def test_workflow_uses_mcp_retrieval_client_when_configured(tmp_path: Path):
    product_store = LocalFaissStore(tmp_path / "product")
    support_store = LocalFaissStore(tmp_path / "support")
    product_hit = {
        "text": "Acme Cloud Backup is the supported product.",
        "source": "product.pdf",
        "page": 1,
        "kind": "manual",
        "score": 0.8,
    }
    support_hit = {
        "text": "Eligible refunds arrive within five business days.",
        "source": "refund_policy.pdf",
        "page": 2,
        "kind": "policy",
        "score": 0.9,
    }
    retrieval = FakeRetrievalClient([product_hit], [support_hit])
    workflow = SupportWorkflow(
        query_agent=FakeQueryAgent(),
        knowledge_agent=FakeKnowledgeAgent(),
        resolution_agent=FakeResolutionAgent(0.90),
        product_store=product_store,
        support_store=support_store,
        ticket_logger=TicketLogger(tmp_path / "tickets.csv"),
        escalation_mailer=FakeEscalationMailer(),
        retrieval_client=retrieval,
    )

    result = await workflow.process("When will my refund arrive?", "customer-mcp")

    assert result.status == "resolved"
    assert retrieval.calls == ["search_product_scope", "search_support_knowledge"]
    assert result.passages[0]["source"] == "refund_policy.pdf"


async def test_workflow_resolves_at_or_above_threshold_and_updates_memory(tmp_path: Path):
    product_store = _build_store(tmp_path / "product")
    support_store = _build_store(tmp_path / "support")
    logger = TicketLogger(tmp_path / "tickets.csv")
    mailer = FakeEscalationMailer()
    workflow = SupportWorkflow(
        query_agent=FakeQueryAgent(),
        knowledge_agent=FakeKnowledgeAgent(),
        resolution_agent=FakeResolutionAgent(0.70),
        product_store=product_store,
        support_store=support_store,
        ticket_logger=logger,
        escalation_mailer=mailer,
        confidence_threshold=0.70,
    )

    result = await workflow.process("When will my refund arrive?", "customer-7")

    assert result.status == "resolved"
    assert result.resolution.confidence == 0.70
    assert result.customer_message == result.resolution.answer
    assert mailer.calls == []
    memory_records = [record for record in support_store.records if record["kind"] == "auto_resolved_case"]
    assert memory_records
    assert "When will my refund arrive?" not in memory_records[0]["text"]
    assert "Your eligible refund" not in memory_records[0]["text"]
    assert "refund_policy.pdf" in memory_records[0]["text"]


async def test_workflow_escalates_low_confidence_and_does_not_learn_unverified_answer(tmp_path: Path):
    product_store = _build_store(tmp_path / "product")
    support_store = _build_store(tmp_path / "support")
    logger = TicketLogger(tmp_path / "tickets.csv")
    mailer = FakeEscalationMailer()
    workflow = SupportWorkflow(
        query_agent=FakeQueryAgent(),
        knowledge_agent=FakeKnowledgeAgent(),
        resolution_agent=FakeResolutionAgent(0.69),
        product_store=product_store,
        support_store=support_store,
        ticket_logger=logger,
        escalation_mailer=mailer,
        confidence_threshold=0.70,
    )

    result = await workflow.process("When will my refund arrive?", "customer-7")

    assert result.status == "escalated"
    assert "confidence" in result.escalation_reason.lower()
    assert "five business days" not in result.customer_message.lower()
    assert "human support" in result.customer_message.lower()
    assert len(mailer.calls) == 1
    assert not any(record["kind"] == "auto_resolved_case" for record in support_store.records)


async def test_workflow_escalates_high_priority_even_with_confident_answer(tmp_path: Path):
    product_store = _build_store(tmp_path / "product")
    support_store = _build_store(tmp_path / "support")
    mailer = FakeEscalationMailer()
    workflow = SupportWorkflow(
        query_agent=FakeQueryAgent(priority="high"),
        knowledge_agent=FakeKnowledgeAgent(),
        resolution_agent=FakeResolutionAgent(0.95),
        product_store=product_store,
        support_store=support_store,
        ticket_logger=TicketLogger(tmp_path / "tickets.csv"),
        escalation_mailer=mailer,
        confidence_threshold=0.70,
    )

    result = await workflow.process("I will report this fraudulent charge", "customer-8")

    assert result.status == "escalated"
    assert "priority" in result.escalation_reason.lower()
    assert len(mailer.calls) == 1


async def test_logging_failure_prevents_untracked_escalation_email(tmp_path: Path):
    product_store = _build_store(tmp_path / "product")
    support_store = _build_store(tmp_path / "support")
    mailer = FakeEscalationMailer()
    workflow = SupportWorkflow(
        query_agent=FakeQueryAgent(priority="high"),
        knowledge_agent=FakeKnowledgeAgent(),
        resolution_agent=FakeResolutionAgent(0.95),
        product_store=product_store,
        support_store=support_store,
        ticket_logger=FailingLogger(),
        escalation_mailer=mailer,
    )

    with pytest.raises(OSError, match="disk full"):
        await workflow.process("I will report this fraudulent charge", "customer-8")

    assert len(mailer.calls) == 0


async def test_identical_later_tickets_have_distinct_idempotent_escalation_deliveries(tmp_path: Path):
    product_store = _build_store(tmp_path / "product")
    support_store = _build_store(tmp_path / "support")
    logger = TicketLogger(tmp_path / "tickets.csv")
    mailer = FakeEscalationMailer()
    workflow = SupportWorkflow(
        query_agent=FakeQueryAgent(priority="high"),
        knowledge_agent=FakeKnowledgeAgent(),
        resolution_agent=FakeResolutionAgent(0.95),
        product_store=product_store,
        support_store=support_store,
        ticket_logger=logger,
        escalation_mailer=mailer,
    )

    await workflow.process("I will report this fraudulent charge", "customer-8")
    await workflow.process("I will report this fraudulent charge", "customer-8")

    rows = logger.read_all()
    assert len(mailer.calls) == 2
    assert rows[0]["escalation_email_status"] == "test"
    assert rows[1]["escalation_email_status"] == "test"
    assert rows[0]["escalation_delivery_id"] != rows[1]["escalation_delivery_id"]


async def test_workflow_rejects_fabricated_source_filename(tmp_path: Path):
    product_store = _build_store(tmp_path / "product")
    support_store = _build_store(tmp_path / "support")
    workflow = SupportWorkflow(
        query_agent=FakeQueryAgent(),
        knowledge_agent=FakeKnowledgeAgent(),
        resolution_agent=FabricatedCitationAgent(0.99),
        product_store=product_store,
        support_store=support_store,
        ticket_logger=TicketLogger(tmp_path / "tickets.csv"),
        escalation_mailer=FakeEscalationMailer(),
        confidence_threshold=0.70,
    )

    result = await workflow.process("When will my refund arrive?", "customer-7")

    assert result.status == "escalated"
    assert "citation" in result.escalation_reason.lower()
    assert result.resolution.grounded is False


async def test_scope_gate_overrides_llm_when_no_relevant_product_hit(tmp_path: Path):
    product_store = _build_store(tmp_path / "product")
    support_store = _build_store(tmp_path / "support")
    workflow = SupportWorkflow(
        query_agent=FakeQueryAgent(related=True),
        knowledge_agent=FakeKnowledgeAgent(),
        resolution_agent=FakeResolutionAgent(0.99),
        product_store=product_store,
        support_store=support_store,
        ticket_logger=TicketLogger(tmp_path / "tickets.csv"),
        escalation_mailer=FakeEscalationMailer(),
        confidence_threshold=0.70,
        retrieval_min_score=0.08,
    )

    result = await workflow.process("astronomy weather on mars", "customer-9")

    assert result.status == "escalated"
    assert result.classification.is_product_related is False


def test_pdf_ingestion_rejects_oversized_direct_input(tmp_path: Path):
    product_store = LocalFaissStore(tmp_path / "product")
    support_store = LocalFaissStore(tmp_path / "support")

    with pytest.raises(ValueError, match="maximum allowed size"):
        ingest_pdf(
            b"x" * (MAX_PDF_BYTES + 1),
            filename="oversized.pdf",
            product_store=product_store,
            support_store=support_store,
        )


def test_ticket_logger_updates_human_review_status_atomically(tmp_path: Path):
    logger = TicketLogger(tmp_path / "tickets.csv")
    classification = Classification(
        category="billing",
        urgency="high",
        sentiment="angry",
        priority="high",
        summary="Duplicate charge",
        is_product_related=True,
        rationale="Billing policy matched.",
    )
    resolution = Resolution(
        answer="A reviewer should confirm the duplicate charge.",
        confidence=0.6,
        source_filenames=["billing_policy.pdf"],
        grounded=True,
    )
    logger.append(
        ticket_id="T-REVIEW1",
        user_id="customer-1",
        query="Charged twice",
        classification=classification,
        resolution=resolution,
        status="escalated",
        escalation_reason="Human review required",
    )

    updated = logger.update_ticket(
        "T-REVIEW1",
        status="resolved",
        review_reply="The duplicate charge has been refunded.",
        review_email_status="sent",
    )

    assert updated is True
    row = logger.read_all()[0]
    assert row["status"] == "resolved"
    assert row["review_reply"] == "The duplicate charge has been refunded."
    assert row["review_email_status"] == "sent"
    assert row["reviewed_at_utc"].endswith("Z")


def test_automatic_escalation_email_excludes_customer_query_and_identity(monkeypatch):
    sent_messages = []

    class FakeSmtp:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def starttls(self):
            pass

        def send_message(self, message):
            sent_messages.append(message)

    monkeypatch.setattr("support_system.email_service.smtplib.SMTP", FakeSmtp)
    mailer = EscalationMailer(
        enabled=True,
        host="smtp.example.com",
        sender="support@example.com",
        recipient="support@example.com",
    )

    result = mailer.send(
        ticket_id="T-PRIVATE",
        user_id="customer@example.com",
        category="billing",
        priority="high",
        reason="Human review required",
        query="My secret card number is 4111111111111111",
    )

    body = sent_messages[0].get_content()
    assert result["sent"] is True
    assert "Category: billing" in body
    assert "sensitive details remain in the local audit" in body.lower()
    assert "customer@example.com" not in body
    assert "My secret card" not in body
    assert "4111111111111111" not in body


def test_human_review_email_targets_configured_support_address(monkeypatch):
    sent_messages = []

    class FakeSmtp:
        def __init__(self, host, port, timeout):
            self.host = host
            self.port = port

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def starttls(self):
            return None

        def login(self, username, password):
            return None

        def send_message(self, message):
            sent_messages.append(message)

    monkeypatch.setattr("support_system.email_service.smtplib.SMTP", FakeSmtp)
    mailer = EscalationMailer(
        enabled=True,
        host="smtp.example.com",
        username="support@example.com",
        password="secret",
        sender="support@example.com",
        recipient="support@example.com",
    )

    result = mailer.send_review_reply(
        ticket_id="T-REVIEW1",
        user_id="customer-1",
        original_query="Charged twice",
        reply_text="The duplicate charge has been refunded.",
    )

    assert result["sent"] is True
    assert sent_messages[0]["To"] == "support@example.com"
    assert "The duplicate charge has been refunded." in sent_messages[0].get_content()


def test_review_delivery_claim_is_atomic_and_validates_before_send(tmp_path: Path):
    logger = TicketLogger(tmp_path / "tickets.csv")
    classification = Classification(
        category="billing",
        urgency="high",
        sentiment="angry",
        priority="high",
        summary="Duplicate charge",
        is_product_related=True,
        rationale="Billing policy matched.",
    )
    resolution = Resolution(
        answer="Review this charge.",
        confidence=0.6,
        source_filenames=["billing_policy.pdf"],
        grounded=True,
    )
    logger.append(
        ticket_id="T-CLAIM1",
        user_id="customer-1",
        query="Charged twice",
        classification=classification,
        resolution=resolution,
        status="escalated",
        escalation_reason="Human review required",
    )

    with pytest.raises(ValueError, match="8,000"):
        logger.claim_review_delivery(
            "T-CLAIM1",
            delivery_id="too-large",
            review_reply="x" * 8001,
            final_status="resolved",
        )

    def claim(delivery_id: str) -> bool:
        return logger.claim_review_delivery(
            "T-CLAIM1",
            delivery_id=delivery_id,
            review_reply="Approved response",
            final_status="resolved",
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(claim, ["delivery-a", "delivery-b"]))

    assert sum(results) == 1
    row = logger.read_all()[0]
    assert row["review_email_status"].startswith("sending:")
    claimed_delivery = row["review_delivery_id"]
    assert logger.finalize_review_delivery(
        "T-CLAIM1",
        delivery_id=claimed_delivery,
        sent=False,
        detail="invalid recipient",
    )
    assert logger.claim_review_delivery(
        "T-CLAIM1",
        delivery_id="delivery-retry",
        review_reply="Approved response after correction",
        final_status="resolved",
    )


def test_invalid_review_recipient_returns_failure_instead_of_raising():
    mailer = EscalationMailer(
        enabled=True,
        host="smtp.example.com",
        sender="support@example.com",
        recipient="bad\nrecipient@example.com",
    )

    result = mailer.send_review_reply(
        ticket_id="T-REVIEW1",
        user_id="customer-1",
        original_query="Charged twice",
        reply_text="Approved response",
    )

    assert result["sent"] is False
    assert "configuration" in result["detail"].lower()


def test_legacy_csv_migration_neutralizes_formula_cells(tmp_path: Path):
    path = tmp_path / "tickets.csv"
    legacy_fields = ["ticket_id", "user_id", "query", "status"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=legacy_fields)
        writer.writeheader()
        writer.writerow(
            {
                "ticket_id": "T-OLD1",
                "user_id": "=CMD()",
                "query": "+DANGER",
                "status": "escalated",
            }
        )

    rows = TicketLogger(path).read_all()

    assert rows[0]["user_id"] == "'=CMD()"
    assert rows[0]["query"] == "'+DANGER"
    with path.open(newline="", encoding="utf-8") as handle:
        persisted = list(csv.DictReader(handle))[0]
    assert persisted["user_id"] == "'=CMD()"
    assert persisted["query"] == "'+DANGER"

