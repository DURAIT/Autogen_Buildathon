from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from support_system.agents import CustomerQueryAgent, KnowledgeBaseAgent, ResolutionAgent
from support_system.config import Settings
from support_system.rag import LocalFaissStore


def test_settings_reject_invalid_thresholds_and_timeout(monkeypatch):
    monkeypatch.setenv("CONFIDENCE_THRESHOLD", "nan")
    with pytest.raises(ValueError, match="CONFIDENCE_THRESHOLD"):
        Settings.from_env()

    monkeypatch.setenv("CONFIDENCE_THRESHOLD", "0.70")
    monkeypatch.setenv("MCP_TIMEOUT_SECONDS", "0")
    with pytest.raises(ValueError, match="MCP_TIMEOUT_SECONDS"):
        Settings.from_env()


def test_autogen_agents_construct_with_openai_compatible_client(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:9999/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-only")
    monkeypatch.setenv("MODEL_NAME", "test-model")
    client = Settings.from_env().build_model_client()
    product = LocalFaissStore(tmp_path / "product")
    support = LocalFaissStore(tmp_path / "support")

    query_agent = CustomerQueryAgent(client, product)
    knowledge_agent = KnowledgeBaseAgent(client, support)
    agents = [
        query_agent.agent.name,
        knowledge_agent.agent.name,
        ResolutionAgent(client).agent.name,
    ]

    assert agents == ["customer_query_agent", "knowledge_base_agent", "resolution_agent"]
    assert query_agent.agent._tools[0].schema["strict"] is True
    assert knowledge_agent.agent._tools[0].schema["strict"] is True


def test_app_wires_pdf_ingestion_to_persistent_data_dir_graph():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    source = app_path.read_text(encoding="utf-8")

    assert 'KnowledgeGraph(Path(settings.data_dir) / "knowledge_graph.json")' in source
    assert "graph=knowledge_graph" in source


def test_streamlit_app_renders_reference_dashboard_without_exception():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(app_path, default_timeout=20).run()

    assert not app.exception
    assert [title.value for title in app.title] == ["Customer Support Analytics"]
    assert app.radio[0].options == [
        "Overview",
        "Customer Queries",
        "Log Analytics",
        "Knowledge Coverage",
        "Admin Console",
        "Settings",
    ]
    metric_labels = [metric.label for metric in app.metric]
    assert "Total Queries Received" in metric_labels
    assert "Auto-resolution Rate" in metric_labels
    assert "Total Logs Collected" in metric_labels
    assert "Knowledge Vectors" in metric_labels
    assert "MCP Tool Server" in metric_labels
    assert any("Operations intelligence" in text.value for text in app.markdown)
    rendered_markdown = "\n".join(text.value for text in app.markdown)
    assert "Agent Pipeline" in rendered_markdown
    assert "Live operations command center" in rendered_markdown
    subheaders = [item.value for item in app.subheader]
    assert subheaders[:4] == [
        "Human support review",
        "Tickets per category",
        "Ticket status",
        "Ticket log",
    ]
