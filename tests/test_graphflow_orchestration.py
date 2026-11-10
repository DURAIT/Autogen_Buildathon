"""Tests for the AutoGen GraphFlow topology used to wire the real production agents.

These tests exercise `support_system.agents.build_support_graphflow`, which wires the
three real, LLM-backed agents (CustomerQueryAgent, KnowledgeBaseAgent, ResolutionAgent)
into an explicit linear AutoGen `GraphFlow`/`DiGraphBuilder` graph with an explicit
termination condition. No real network/LLM call is made: a `ReplayChatCompletionClient`
(shipped with the installed autogen-ext==0.7.5) supplies canned, pre-validated
structured JSON responses for each of the three agent turns.
"""

from __future__ import annotations

import json

import pytest
from autogen_core.models import ModelFamily, ModelInfo
from autogen_ext.models.replay import ReplayChatCompletionClient

from support_system.agents import (
    CustomerQueryAgent,
    KnowledgeBaseAgent,
    KnowledgeSynthesis,
    ResolutionAgent,
    build_support_graphflow,
)
from support_system.models import Classification, Resolution
from support_system.rag import LocalFaissStore

MODEL_INFO = ModelInfo(
    vision=False,
    function_calling=False,
    json_output=True,
    family=ModelFamily.UNKNOWN,
    structured_output=True,
)


def _canned_responses() -> list[str]:
    classification = Classification(
        category="refund",
        urgency="moderate",
        sentiment="frustrated",
        priority="moderate",
        summary="Refund delayed beyond the promised window.",
        is_product_related=True,
        rationale="Matched refund policy.",
    )
    synthesis = KnowledgeSynthesis(
        relevant=True,
        summary="Refunds are issued within five business days.",
        source_filenames=["refund_policy.pdf"],
        ranked_chunk_ids=[],
    )
    resolution = Resolution(
        answer="Your eligible refund will arrive within five business days.",
        confidence=0.9,
        source_filenames=["refund_policy.pdf"],
        grounded=True,
    )
    return [
        classification.model_dump_json(),
        synthesis.model_dump_json(),
        resolution.model_dump_json(),
    ]


def _build_team(tmp_path):
    client = ReplayChatCompletionClient(_canned_responses(), model_info=MODEL_INFO)
    store = LocalFaissStore(tmp_path / "kb")
    query_agent = CustomerQueryAgent(client, store, use_local_tools=False)
    knowledge_agent = KnowledgeBaseAgent(client, store, use_local_tools=False)
    resolution_agent = ResolutionAgent(client)
    team = build_support_graphflow(query_agent, knowledge_agent, resolution_agent)
    return team, query_agent, knowledge_agent, resolution_agent


def test_graphflow_contains_exactly_the_three_llm_agent_nodes(tmp_path):
    """The GraphFlow team's participants are exactly the three LLM-backed agent nodes."""
    team, query_agent, knowledge_agent, resolution_agent = _build_team(tmp_path)

    participant_names = {participant.name for participant in team._participants}

    assert participant_names == {
        query_agent.agent.name,
        knowledge_agent.agent.name,
        resolution_agent.agent.name,
    }
    assert "log_agent" not in participant_names


def test_graphflow_edges_enforce_linear_query_to_knowledge_to_resolution_order(tmp_path):
    """Edges form a simple linear DAG: query -> knowledge -> resolution, not round robin."""
    team, query_agent, knowledge_agent, resolution_agent = _build_team(tmp_path)

    graph = team._graph
    query_name = query_agent.agent.name
    knowledge_name = knowledge_agent.agent.name
    resolution_name = resolution_agent.agent.name

    query_targets = {edge.target for edge in graph.nodes[query_name].edges}
    knowledge_targets = {edge.target for edge in graph.nodes[knowledge_name].edges}
    resolution_targets = {edge.target for edge in graph.nodes[resolution_name].edges}

    assert query_targets == {knowledge_name}
    assert knowledge_targets == {resolution_name}
    assert resolution_targets == set()  # Terminal node: no outgoing edges, no looping.


async def test_graphflow_run_executes_each_agent_exactly_once_in_order(tmp_path):
    """Running the graph with canned LLM responses visits each agent exactly once, in order."""
    team, query_agent, knowledge_agent, resolution_agent = _build_team(tmp_path)

    result = await team.run(task="Customer query: When will my delayed refund arrive?")

    agent_sources = [
        message.source
        for message in result.messages
        if message.source != "user"
    ]

    assert agent_sources == [
        query_agent.agent.name,
        knowledge_agent.agent.name,
        resolution_agent.agent.name,
    ]


async def test_graphflow_terminates_after_resolution_agent_single_turn(tmp_path):
    """The run halts right after the resolution agent's one turn -- no loop, no extra turns."""
    team, query_agent, knowledge_agent, resolution_agent = _build_team(tmp_path)

    result = await team.run(task="Customer query: When will my delayed refund arrive?")

    resolution_messages = [
        message for message in result.messages if message.source == resolution_agent.agent.name
    ]
    assert len(resolution_messages) == 1
    assert result.stop_reason is not None

    classification_payload = json.loads(result.messages[1].to_text())
    assert classification_payload["category"] == "refund"
