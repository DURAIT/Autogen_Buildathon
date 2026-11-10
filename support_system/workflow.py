from __future__ import annotations

import hashlib
import math
import re
import time
from uuid import uuid4

from .agents import LogAgent
from .models import WorkflowResult

HIGH_RISK_PATTERN = re.compile(
    r"\b(fraud|fraudulent|chargeback|legal|lawyer|lawsuit|police|stolen|unsafe|injury|data loss)\b",
    re.IGNORECASE,
)


class SupportWorkflow:
    """Explicit sequential orchestration with deterministic safety and grounding gates.

    Orchestration-mechanism note (see also support_system/agents.py::build_support_graphflow
    and README.md "Architecture"): the three LLM-backed roles (query_agent, knowledge_agent,
    resolution_agent) are still invoked here with direct, ordered Python calls rather than by
    driving an AutoGen GraphFlow team inline. AutoGen's `GraphFlow`/`DiGraphBuilder` (with an
    explicit linear query->knowledge->resolution edge set and a `SourceMatchTermination` that
    halts the run right after the resolution agent's single turn) is implemented and exercised
    in `build_support_graphflow`, proving the topology/termination AutoGen provides for the real
    `AssistantAgent`-backed agents app.py constructs. It is intentionally NOT called from
    `process()` because every existing and new fake-agent unit test injects plain Python
    doubles (FakeQueryAgent/FakeKnowledgeAgent/FakeResolutionAgent/...) that are not AutoGen
    ChatAgents and cannot be scheduled inside a GraphFlow team. Calling `.classify()`,
    `.retrieve()`/`.synthesize()`, and `.resolve()` directly keeps this method duck-type
    compatible with both the fakes and the real agents, while still guaranteeing the exact same
    fixed order GraphFlow's edges would enforce. All deterministic safety/grounding gates below
    (high-risk regex override, product-scope gate, citation/grounding validation, confidence
    threshold, escalation reasons, ticket logging, escalation email claim/send, and the
    memory-write gate) run in plain Python, unchanged, strictly after the three agents'
    structured outputs are produced -- regardless of which mechanism produced them.
    """

    def __init__(
        self,
        *,
        query_agent,
        knowledge_agent,
        resolution_agent,
        product_store,
        support_store,
        ticket_logger,
        escalation_mailer,
        confidence_threshold: float = 0.70,
        retrieval_min_score: float = 0.20,
        retrieval_client=None,
        perf_counter=None,
    ) -> None:
        if not 0 <= confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if not -1 <= retrieval_min_score <= 1:
            raise ValueError("retrieval_min_score must be between -1 and 1")
        self.query_agent = query_agent
        self.knowledge_agent = knowledge_agent
        self.resolution_agent = resolution_agent
        self.product_store = product_store
        self.support_store = support_store
        self.log_agent = LogAgent(ticket_logger, support_store)
        self.ticket_logger = ticket_logger
        self.escalation_mailer = escalation_mailer
        self.confidence_threshold = confidence_threshold
        self.retrieval_min_score = retrieval_min_score
        self.retrieval_client = retrieval_client
        self.perf_counter = perf_counter or time.perf_counter

    def _elapsed_ms(self, started_at: float) -> float:
        elapsed = (self.perf_counter() - started_at) * 1000.0
        if not math.isfinite(elapsed) or elapsed < 0:
            raise RuntimeError("perf_counter must produce finite, non-decreasing values")
        return elapsed

    async def process(self, query: str, user_id: str) -> WorkflowResult:
        total_started_at = self.perf_counter()
        timings_ms: dict[str, float] = {}
        clean_query = " ".join(query.split())
        clean_user_id = " ".join(user_id.split()) or "anonymous"
        if len(clean_query) < 3:
            raise ValueError("Please enter a meaningful customer query.")
        if len(clean_query) > 8000:
            raise ValueError("Customer query is too long (maximum 8,000 characters).")

        ticket_id = f"T-{uuid4().hex[:10].upper()}"
        stage_started_at = self.perf_counter()
        if self.retrieval_client is not None:
            product_hits = await self.retrieval_client.search_product_scope(clean_query, top_k=4)
        else:
            product_hits = self.product_store.search(
                clean_query, top_k=4, min_score=self.retrieval_min_score
            )
        timings_ms["product_retrieval"] = self._elapsed_ms(stage_started_at)

        stage_started_at = self.perf_counter()
        classification = await self.query_agent.classify(clean_query, product_hits)
        timings_ms["classification"] = self._elapsed_ms(stage_started_at)

        # The LLM can make classification suggestions, but it cannot bypass these gates.
        if not product_hits:
            classification.is_product_related = False
            classification.rationale += " Deterministic scope retrieval found no relevant product match."
        if HIGH_RISK_PATTERN.search(clean_query):
            classification.priority = "high"
            classification.urgency = "high"
            classification.rationale += " Deterministic high-risk phrase detected."

        stage_started_at = self.perf_counter()
        if self.retrieval_client is not None:
            retrieval_query = f"{classification.category} {classification.summary} {clean_query}"
            passages = await self.retrieval_client.search_support_knowledge(
                retrieval_query, top_k=6
            )
            passages = await self.knowledge_agent.synthesize(
                clean_query, classification, passages
            )
        else:
            passages = await self.knowledge_agent.retrieve(
                clean_query, classification, self.support_store
            )
        passages = [
            passage
            for passage in passages
            if float(passage.get("score", -1.0)) >= self.retrieval_min_score
        ]
        timings_ms["support_retrieval_synthesis"] = self._elapsed_ms(stage_started_at)

        stage_started_at = self.perf_counter()
        resolution = await self.resolution_agent.resolve(clean_query, classification, passages)
        timings_ms["resolution"] = self._elapsed_ms(stage_started_at)

        reasons: list[str] = []
        if not classification.is_product_related:
            reasons.append("Query is outside the supported product scope")
        if classification.priority == "high":
            reasons.append("High-priority case requires human review")
        if not passages:
            reasons.append("No relevant knowledge-base passages met the retrieval threshold")

        synthesis = getattr(self.knowledge_agent, "last_synthesis", None)
        if synthesis is not None and not synthesis.relevant:
            reasons.append("Knowledge agent marked the retrieved evidence as irrelevant")

        retrieved_sources = {str(item.get("source")) for item in passages if item.get("source")}
        cited_sources = set(resolution.source_filenames)
        citations_valid = bool(cited_sources) and cited_sources.issubset(retrieved_sources)
        if not citations_valid:
            resolution.grounded = False
            reasons.append("Response citations do not match retrieved evidence")
        if not resolution.grounded:
            reasons.append("Generated response is not sufficiently grounded")
        if resolution.confidence < self.confidence_threshold:
            reasons.append(
                f"Response confidence {resolution.confidence:.0%} is below "
                f"the {self.confidence_threshold:.0%} threshold"
            )

        # Preserve order while removing duplicate reasons.
        reasons = list(dict.fromkeys(reasons))
        status = "escalated" if reasons else "resolved"
        escalation_reason = "; ".join(reasons)
        customer_message = (
            resolution.answer
            if status == "resolved"
            else (
                f"Thank you for contacting support. Your request needs human review and has been "
                f"escalated as ticket {ticket_id}. A human support specialist will follow up."
            )
        )
        email_status = "not_required"
        escalation_delivery_id = ""
        escalation_delivery_state = ""
        if status == "escalated":
            escalation_delivery_id = hashlib.sha256(
                f"escalation-email\0{ticket_id}".encode("utf-8")
            ).hexdigest()[:24]
            escalation_delivery_state = "pending"
            email_status = "pending"

        # Persist the ticket before any external email side effect.
        stage_started_at = self.perf_counter()
        self.log_agent.record(
            ticket_id=ticket_id,
            user_id=clean_user_id,
            query=clean_query,
            classification=classification,
            resolution=resolution,
            status=status,
            escalation_reason=escalation_reason,
            escalation_delivery_id=escalation_delivery_id,
            escalation_delivery_state=escalation_delivery_state,
            escalation_email_status=email_status,
        )
        timings_ms["local_persistence"] = self._elapsed_ms(stage_started_at)

        stage_started_at = self.perf_counter()
        if status == "escalated":
            claimed = self.ticket_logger.claim_escalation_delivery(
                ticket_id, escalation_delivery_id
            )
            if not claimed:
                self.ticket_logger.mark_escalation_duplicate(ticket_id)
                email_status = "duplicate_suppressed"
            else:
                try:
                    email_result = self.escalation_mailer.send(
                        ticket_id=ticket_id,
                        category=classification.category,
                        priority=classification.priority,
                        reason=escalation_reason,
                    )
                    sent = bool(email_result.get("sent"))
                    email_status = str(email_result.get("detail", "unknown"))
                except Exception as exc:
                    sent = False
                    email_status = f"Escalation email failed: {type(exc).__name__}"
                self.ticket_logger.finalize_escalation_delivery(
                    ticket_id,
                    sent=sent,
                    detail=email_status,
                )
        timings_ms["escalation_email_handling"] = self._elapsed_ms(stage_started_at)

        if status == "resolved":
            self.log_agent.remember_resolved_case(
                ticket_id=ticket_id,
                query=clean_query,
                classification=classification,
                resolution=resolution,
            )

        resolution_stage_output = (
            resolution.model_dump()
            if status == "resolved"
            else {
                "confidence": resolution.confidence,
                "grounded": resolution.grounded,
                "source_filenames": resolution.source_filenames,
                "draft_withheld": True,
            }
        )
        stages = [
            {"agent": "Customer Query Agent", "status": "complete", "output": classification.model_dump()},
            {"agent": "Knowledge Base Agent", "status": "complete", "output": {"passages": len(passages)}},
            {"agent": "Resolution Agent", "status": "complete", "output": resolution_stage_output},
            {
                "agent": "Log Agent",
                "status": "complete",
                "output": {"ticket_id": ticket_id, "decision": status},
            },
        ]
        total_latency_ms = self._elapsed_ms(total_started_at)
        if not self.ticket_logger.update_observability(
            ticket_id,
            timings_ms,
            total_latency_ms,
        ):
            raise RuntimeError("ticket disappeared before observability could be persisted")
        result_timings = {**timings_ms, "total_latency_ms": total_latency_ms}
        return WorkflowResult(
            ticket_id=ticket_id,
            user_id=clean_user_id,
            status=status,
            classification=classification,
            resolution=resolution,
            customer_message=customer_message,
            escalation_reason=escalation_reason,
            email_status=email_status,
            passages=passages,
            stages=stages,
            timings_ms=result_timings,
        )
