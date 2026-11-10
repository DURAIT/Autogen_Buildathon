"""Sample customer-complaint scenarios covering the documented workflow decision paths.

These six scenarios are illustrative sample complaints inferred from the documented
workflow decision paths (README.md architecture section); replace with the official
buildathon complaint list if one exists.

Decision paths covered (see README.md "Architecture"):
1. Resolved at/above 70% confidence.
2. Escalated for low confidence (below the 70% threshold).
3. Escalated for a high-priority/safety keyword (deterministic regex override).
4. Escalated for out-of-product-scope (no relevant product-scope retrieval hit).
5. Escalated for citation/grounding mismatch (cited sources not in retrieved evidence).
6. Resolved and correctly written to long-term memory (auto_resolved_case marker).

Plus two additional author-chosen scenarios covering different gate combinations.
"""

from __future__ import annotations

from pathlib import Path

from support_system.models import Classification, Resolution
from support_system.rag import LocalFaissStore
from support_system.storage import TicketLogger
from support_system.workflow import SupportWorkflow
from tests.test_support_system import (
    FakeEscalationMailer,
    FakeKnowledgeAgent,
    FakeQueryAgent,
    FakeResolutionAgent,
    FabricatedCitationAgent,
)


def _seeded_store(path: Path, text: str, source: str, kind: str = "policy") -> LocalFaissStore:
    store = LocalFaissStore(path)
    store.add_texts([text], [{"source": source, "page": 1, "kind": kind}])
    return store


class ScopedQueryAgent(FakeQueryAgent):
    """Classifies as product-related; used when the scope gate must be driven by retrieval."""


async def test_resolved_at_or_above_seventy_percent_confidence(tmp_path: Path):
    """Decision path: resolved because confidence is at/above the 70% threshold."""
    product_store = _seeded_store(
        tmp_path / "product",
        "Acme Cloud Backup subscription and billing settings are managed in Account Settings.",
        "product.pdf",
    )
    support_store = _seeded_store(
        tmp_path / "support",
        "You can cancel your subscription anytime before your next billing date with no fee.",
        "cancellation_policy.pdf",
    )
    workflow = SupportWorkflow(
        query_agent=FakeQueryAgent(priority="low"),
        knowledge_agent=FakeKnowledgeAgent(),
        resolution_agent=FakeResolutionAgent(0.85),
        product_store=product_store,
        support_store=support_store,
        ticket_logger=TicketLogger(tmp_path / "tickets.csv"),
        escalation_mailer=FakeEscalationMailer(),
        confidence_threshold=0.70,
        retrieval_min_score=0.05,
    )

    result = await workflow.process(
        "I want to cancel my subscription before my next billing date, how do I do that?",
        "customer-101",
    )

    assert result.status == "resolved"
    assert result.resolution.source_filenames == ["cancellation_policy.pdf"]
    assert result.passages[0]["source"] == "cancellation_policy.pdf"


async def test_escalated_for_low_confidence_below_threshold(tmp_path: Path):
    """Decision path: escalated because resolution confidence is below the 70% threshold."""
    product_store = _seeded_store(
        tmp_path / "product",
        "Acme Cloud Backup annual plan billing and refunds are supported product features.",
        "product.pdf",
    )
    support_store = _seeded_store(
        tmp_path / "support",
        "Partial refunds for your annual plan are evaluated case-by-case by a billing specialist.",
        "billing_policy.pdf",
    )
    workflow = SupportWorkflow(
        query_agent=FakeQueryAgent(priority="low"),
        knowledge_agent=FakeKnowledgeAgent(),
        resolution_agent=FakeResolutionAgent(0.55),
        product_store=product_store,
        support_store=support_store,
        ticket_logger=TicketLogger(tmp_path / "tickets.csv"),
        escalation_mailer=FakeEscalationMailer(),
        confidence_threshold=0.70,
        retrieval_min_score=0.05,
    )

    result = await workflow.process(
        "I was only partially refunded for my annual plan and I don't understand why.",
        "customer-102",
    )

    assert result.status == "escalated"
    assert "confidence" in result.escalation_reason.lower()
    assert result.resolution.source_filenames == ["billing_policy.pdf"]


async def test_escalated_for_high_priority_safety_keyword(tmp_path: Path):
    """Decision path: escalated because a deterministic high-risk/safety keyword was detected."""
    product_store = _seeded_store(
        tmp_path / "product", "Acme Smart Plug firmware updates ship over Wi-Fi.", "product.pdf"
    )
    support_store = _seeded_store(
        tmp_path / "support",
        "Report any device malfunction immediately and stop use.",
        "safety_policy.pdf",
    )
    workflow = SupportWorkflow(
        query_agent=FakeQueryAgent(priority="low"),
        knowledge_agent=FakeKnowledgeAgent(),
        resolution_agent=FakeResolutionAgent(0.95),
        product_store=product_store,
        support_store=support_store,
        ticket_logger=TicketLogger(tmp_path / "tickets.csv"),
        escalation_mailer=FakeEscalationMailer(),
        confidence_threshold=0.70,
        retrieval_min_score=0.05,
    )

    result = await workflow.process(
        "Your smart plug overheated and caused an injury to my hand, this is extremely unsafe!",
        "customer-103",
    )

    assert result.status == "escalated"
    assert "priority" in result.escalation_reason.lower()
    assert result.resolution.source_filenames == ["safety_policy.pdf"]


async def test_escalated_for_out_of_product_scope(tmp_path: Path):
    """Decision path: escalated because the query has no relevant product-scope retrieval hit."""
    product_store = _seeded_store(
        tmp_path / "product", "Acme Cloud Backup supports automatic nightly syncs.", "product.pdf"
    )
    support_store = _seeded_store(
        tmp_path / "support",
        "Refunds for eligible purchases are processed within five business days.",
        "refund_policy.pdf",
    )
    workflow = SupportWorkflow(
        query_agent=FakeQueryAgent(priority="low", related=True),
        knowledge_agent=FakeKnowledgeAgent(),
        resolution_agent=FakeResolutionAgent(0.95),
        product_store=product_store,
        support_store=support_store,
        ticket_logger=TicketLogger(tmp_path / "tickets.csv"),
        escalation_mailer=FakeEscalationMailer(),
        confidence_threshold=0.70,
        retrieval_min_score=0.90,  # Deliberately high so the unrelated query misses scope.
    )

    result = await workflow.process(
        "Can you recommend a good recipe for banana bread?",
        "customer-104",
    )

    assert result.status == "escalated"
    assert "scope" in result.escalation_reason.lower()
    assert result.classification.is_product_related is False


async def test_escalated_for_citation_grounding_mismatch(tmp_path: Path):
    """Decision path: escalated because the cited source filename is not among retrieved evidence."""
    product_store = _seeded_store(
        tmp_path / "product", "Acme Cloud Backup is the supported product.", "product.pdf"
    )
    support_store = _seeded_store(
        tmp_path / "support",
        "Password resets are sent to the verified account email within minutes.",
        "account_policy.pdf",
    )
    workflow = SupportWorkflow(
        query_agent=FakeQueryAgent(priority="low"),
        knowledge_agent=FakeKnowledgeAgent(),
        resolution_agent=FabricatedCitationAgent(0.99),
        product_store=product_store,
        support_store=support_store,
        ticket_logger=TicketLogger(tmp_path / "tickets.csv"),
        escalation_mailer=FakeEscalationMailer(),
        confidence_threshold=0.70,
        retrieval_min_score=0.05,
    )

    result = await workflow.process(
        "I never received my password reset email, can you check what happened?",
        "customer-105",
    )

    assert result.status == "escalated"
    assert "citation" in result.escalation_reason.lower()
    assert result.resolution.source_filenames == ["invented-policy.pdf"]
    assert result.passages[0]["source"] == "account_policy.pdf"


async def test_resolved_case_is_written_to_long_term_memory(tmp_path: Path):
    """Decision path: resolved, grounded, in-scope case is written to long-term memory."""
    product_store = _seeded_store(
        tmp_path / "product", "Acme Cloud Backup is the supported product.", "product.pdf"
    )
    support_store = _seeded_store(
        tmp_path / "support",
        "Storage plan upgrades take effect immediately after payment confirmation.",
        "plan_policy.pdf",
    )
    workflow = SupportWorkflow(
        query_agent=FakeQueryAgent(priority="low"),
        knowledge_agent=FakeKnowledgeAgent(),
        resolution_agent=FakeResolutionAgent(0.92),
        product_store=product_store,
        support_store=support_store,
        ticket_logger=TicketLogger(tmp_path / "tickets.csv"),
        escalation_mailer=FakeEscalationMailer(),
        confidence_threshold=0.70,
        retrieval_min_score=0.05,
    )

    result = await workflow.process(
        "I just upgraded my storage plan, when does the extra space become available?",
        "customer-106",
    )

    assert result.status == "resolved"
    assert result.resolution.source_filenames == ["plan_policy.pdf"]
    memory_records = [
        record for record in support_store.records if record["kind"] == "auto_resolved_case"
    ]
    assert memory_records
    assert "plan_policy.pdf" in memory_records[0]["text"]


async def test_escalated_when_no_knowledge_base_passages_meet_retrieval_threshold(tmp_path: Path):
    """Author-added: escalated because no support knowledge-base passage met the score threshold."""
    product_store = _seeded_store(
        tmp_path / "product", "Acme Cloud Backup is the supported product.", "product.pdf"
    )
    support_store = _seeded_store(
        tmp_path / "support",
        "Totally unrelated onboarding tutorial content about account creation.",
        "onboarding.pdf",
    )
    workflow = SupportWorkflow(
        query_agent=FakeQueryAgent(priority="low"),
        knowledge_agent=FakeKnowledgeAgent(),
        resolution_agent=FakeResolutionAgent(0.95),
        product_store=product_store,
        support_store=support_store,
        ticket_logger=TicketLogger(tmp_path / "tickets.csv"),
        escalation_mailer=FakeEscalationMailer(),
        confidence_threshold=0.70,
        retrieval_min_score=0.99,  # Nothing can score this high; forces an empty passage set.
    )

    result = await workflow.process(
        "My backup job has been stuck at 42 percent for three hours, what is going on?",
        "customer-107",
    )

    assert result.status == "escalated"
    assert "retrieval threshold" in result.escalation_reason.lower()
    assert result.resolution.source_filenames == []
    assert result.passages == []


async def test_escalated_for_high_priority_even_when_otherwise_resolvable(tmp_path: Path):
    """Author-added: a legal/fraud-flagged billing dispute escalates despite strong evidence/confidence."""
    product_store = _seeded_store(
        tmp_path / "product", "Acme Cloud Backup is the supported product.", "product.pdf"
    )
    support_store = _seeded_store(
        tmp_path / "support",
        "Disputed charges are reviewed by billing within two business days.",
        "dispute_policy.pdf",
    )
    workflow = SupportWorkflow(
        query_agent=FakeQueryAgent(priority="moderate"),
        knowledge_agent=FakeKnowledgeAgent(),
        resolution_agent=FakeResolutionAgent(0.97),
        product_store=product_store,
        support_store=support_store,
        ticket_logger=TicketLogger(tmp_path / "tickets.csv"),
        escalation_mailer=FakeEscalationMailer(),
        confidence_threshold=0.70,
        retrieval_min_score=0.05,
    )

    result = await workflow.process(
        "I'm seeing a charge on my card I never authorized -- this looks like fraud and I will "
        "involve my lawyer if it isn't reversed.",
        "customer-108",
    )

    assert result.status == "escalated"
    assert "priority" in result.escalation_reason.lower()
    assert result.classification.priority == "high"
    assert result.resolution.source_filenames == ["dispute_policy.pdf"]
