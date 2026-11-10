from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_admin_console_exposes_operational_sections_without_live_side_effects():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(app_path, default_timeout=20).run()

    assert not app.exception
    assert "Admin Console" in app.radio[0].options

    app.radio[0].set_value("Admin Console").run()

    assert not app.exception
    subheaders = [item.value for item in app.subheader]
    assert "Ticket operations" in subheaders
    assert "Latency and status" in subheaders
    assert "Retrieval inspector" in subheaders
    assert "Knowledge graph viewer" in subheaders
    assert "Google Sheets synchronization" in subheaders
    rendered = "\n".join(item.value for item in app.markdown)
    assert "Local administrator surface" in rendered
    assert any("localhost only" in item.value.lower() for item in app.warning)
