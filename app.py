from __future__ import annotations

import asyncio
import html
import json
import logging
import re
from pathlib import Path
from uuid import uuid4

import altair as alt
import pandas as pd
import streamlit as st

from support_system.agents import CustomerQueryAgent, KnowledgeBaseAgent, ResolutionAgent
from support_system.config import Settings
from support_system.ingestion import ingest_pdf
from support_system.knowledge_graph import KnowledgeGraph
from support_system.mcp_client import SupportMcpClient, safe_mcp_url_for_display
from support_system.rag import LocalFaissStore
from support_system.sheets_mirror import (
    GoogleApiSheetsTransport,
    GoogleSheetsMirror,
    SheetsSynchronizer,
    TicketOutbox,
)
from support_system.storage import TicketLogger
from support_system.workflow import SupportWorkflow

logger = logging.getLogger(__name__)

st.set_page_config(page_title="SupportIQ Analytics", page_icon="💬", layout="wide")
st.markdown(
    """
    <style>
    :root {
        --emerald:#10b981;--emerald-dark:#087a5b;--emerald-soft:#e7f8f1;
        --blue:#2563eb;--purple:#7c3aed;--amber:#f59e0b;--danger:#dc4545;
        --ink:#10231d;--muted:#61736d;--line:#d9e5e0;--surface:#ffffff;--canvas:#f2f7f5;
    }
    html,body,[class*="css"]{font-family:'Segoe UI',Roboto,Arial,sans-serif;}
    .stApp{background:var(--canvas);color:var(--ink)}
    [data-testid="stHeader"]{background:rgba(242,247,245,.94);border-bottom:1px solid rgba(16,185,129,.08)}
    [data-testid="stSidebar"]{background:#071c16;border-right:1px solid #174136;color:#f4fff9}
    [data-testid="stSidebar"]>div:first-child{padding-top:1rem}
    [data-testid="stSidebar"] p,[data-testid="stSidebar"] span,[data-testid="stSidebar"] label{color:#eefaf5!important}
    .block-container{max-width:none;padding:1.25rem 1.65rem 3rem}
    h1{font-size:1.72rem!important;letter-spacing:-.04em!important;margin-bottom:.12rem!important;color:#0b231b}
    h2{font-size:1.05rem!important;letter-spacing:-.02em;color:#15382d} h3{font-size:.92rem!important}
    .brand{display:flex;align-items:center;gap:.7rem;padding:.15rem .25rem 1rem}
    .brand-mark{width:36px;height:36px;border-radius:10px;background:var(--emerald);
                display:grid;place-items:center;color:#05291f;font-size:16px;font-weight:900;
                box-shadow:0 0 0 5px rgba(16,185,129,.12)}
    .brand-name{font-size:1.02rem;font-weight:750;color:#ffffff;line-height:1.05}
    .brand-sub{font-size:.7rem;color:#8bb9a9;margin-top:.2rem;letter-spacing:.035em}
    .pipeline-title,.section-kicker{font-size:.64rem;font-weight:800;letter-spacing:.14em;text-transform:uppercase}
    .pipeline-title{color:#72a895;margin:1rem 0 .65rem}
    .section-kicker{color:var(--emerald-dark);margin:1.1rem 0 .45rem}
    .agent-card{display:grid;grid-template-columns:28px 1fr;gap:.5rem;background:#0d2a21;
                border:1px solid #1a4639;border-radius:10px;padding:.66rem .62rem;margin:.42rem 0;
                transition:border-color .16s ease,background .16s ease}
    .agent-card:hover{background:#103428;border-color:#28705b}
    .agent-step{display:grid;place-items:center;width:23px;height:23px;border-radius:7px;
                background:var(--emerald);color:#05291f;font-size:.7rem;font-weight:900}
    .agent-name{font-size:.76rem;font-weight:700;color:#fff}.agent-desc{font-size:.65rem;color:#9ac2b5;margin-top:.18rem;line-height:1.4}
    .page-subtitle{font-size:.84rem;color:var(--muted);margin-bottom:.55rem}
    .command-bar{display:flex;align-items:center;justify-content:space-between;gap:1rem;background:#0b2a21;
                 color:#f4fff9;border:1px solid #164a3b;border-radius:12px;padding:.68rem .85rem;margin:.3rem 0 1rem}
    .command-title{font-size:.76rem;font-weight:750}.command-meta{font-size:.68rem;color:#a7cec1;text-align:right}
    .live-dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:#3ee6a8;margin-right:.45rem;
              box-shadow:0 0 0 4px rgba(62,230,168,.14)}
    div[data-testid="stMetric"]{background:var(--surface);border:1px solid var(--line);border-radius:12px;
                                padding:.9rem 1rem;box-shadow:0 4px 18px rgba(17,52,41,.055);min-height:112px}
    div[data-testid="stMetricLabel"]{font-size:.75rem;color:#536862;font-weight:650;letter-spacing:.01em}
    div[data-testid="stMetricValue"]{font-size:1.86rem;color:#123e31;font-weight:720;letter-spacing:-.045em}
    div[data-testid="stHorizontalBlock"]:has(div[data-testid="stMetric"])>div:nth-child(1) div[data-testid="stMetric"]{border-top:3px solid var(--blue)}
    div[data-testid="stHorizontalBlock"]:has(div[data-testid="stMetric"])>div:nth-child(2) div[data-testid="stMetric"]{border-top:3px solid var(--emerald)}
    div[data-testid="stHorizontalBlock"]:has(div[data-testid="stMetric"])>div:nth-child(3) div[data-testid="stMetric"]{border-top:3px solid var(--amber)}
    div[data-testid="stHorizontalBlock"]:has(div[data-testid="stMetric"])>div:nth-child(4) div[data-testid="stMetric"]{border-top:3px solid var(--purple)}
    div[data-testid="stVerticalBlockBorderWrapper"]{background:var(--surface);border-color:var(--line)!important;
                                                      border-radius:12px;box-shadow:0 4px 18px rgba(17,52,41,.045)}
    div[role="radiogroup"]{gap:.2rem}
    div[role="radiogroup"] label{padding:.55rem .65rem;border-radius:8px;margin:.04rem 0;color:#b9d4cb!important}
    div[role="radiogroup"] label:hover{background:#0e3025}
    div[role="radiogroup"] label:has(input:checked){background:#123f31;color:#ffffff!important;font-weight:700;box-shadow:inset 3px 0 0 var(--emerald)}
    div[role="radiogroup"] label>div:first-child{display:none}
    .status-resolved{padding:.8rem 1rem;border-radius:9px;background:#e9f8f3;color:#08795f;border:1px solid #bcebdc}
    .status-escalated{padding:.8rem 1rem;border-radius:9px;background:#fff1ed;color:#b33a26;border:1px solid #ffd2c8}
    .empty-state{height:235px;display:grid;place-items:center;text-align:center;color:#71837d;
                 background:#f7faf9;border:1px dashed #c7d8d2;border-radius:10px}
    .source-pill{display:inline-block;padding:.25rem .5rem;border-radius:999px;background:var(--emerald-soft);color:var(--emerald-dark);
                 font-size:.72rem;margin:.12rem}
    .stButton>button,.stDownloadButton>button,.stFormSubmitButton>button{border-radius:8px;border:1px solid #c4d6cf;font-weight:700;min-height:2.45rem}
    .stButton>button[kind="primary"],.stFormSubmitButton>button{background:#0b7f5e;border-color:#0b7f5e;color:white}
    .stButton>button[kind="primary"]:hover,.stFormSubmitButton>button:hover{background:#086b50;border-color:#086b50}
    [data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:10px;overflow:hidden}
    .eyebrow{display:inline-flex;align-items:center;gap:.35rem;text-transform:uppercase;letter-spacing:.11em;
             font-size:.64rem;font-weight:800;color:var(--emerald-dark);background:var(--emerald-soft);border:1px solid #bde9d9;
             border-radius:999px;padding:.3rem .58rem;margin:.1rem 0 .45rem}
    div[data-testid="stMetric"]{transition:transform .16s ease,box-shadow .16s ease,border-color .16s ease}
    div[data-testid="stMetric"]:hover{transform:translateY(-2px);border-color:#a9c8bd;box-shadow:0 12px 28px rgba(17,67,50,.10)}
    [data-testid="stSidebar"] [data-testid="stMetric"]{min-height:auto;background:#0d2a21;border-color:#1c4a3c!important;border-top:1px solid #1c4a3c!important}
    [data-testid="stSidebar"] [data-testid="stMetricValue"]{font-size:1.12rem;color:#fff}
    @media (max-width:900px){.block-container{padding:.9rem}.brand-sub,.command-meta{display:none}.command-bar{justify-content:flex-start}}
    @media (prefers-reduced-motion:reduce){*{scroll-behavior:auto!important;transition:none!important}}
    footer{visibility:hidden}
    </style>
    """,
    unsafe_allow_html=True,
)

settings = Settings.from_env()
Path(settings.data_dir).mkdir(parents=True, exist_ok=True)
embedder = settings.build_embedder()
product_store = LocalFaissStore(
    Path(settings.data_dir) / "product_scope_faiss", embedder=embedder
)
support_store = LocalFaissStore(
    Path(settings.data_dir) / "support_knowledge_faiss", embedder=embedder
)
knowledge_graph = KnowledgeGraph(Path(settings.data_dir) / "knowledge_graph.json")
sheets_outbox = (
    TicketOutbox(Path(settings.data_dir) / "google_sheets_outbox.jsonl")
    if settings.google_sheets_configured
    else None
)
ticket_logger = TicketLogger(
    Path(settings.data_dir) / "tickets.csv",
    event_sink=sheets_outbox.enqueue if sheets_outbox else None,
)


def sync_google_sheets(limit: int = 100) -> dict[str, int]:
    """Explicit bounded admin operation; normal ticket processing never calls Google."""
    if sheets_outbox is None:
        return {"sent": 0, "failed": 0, "skipped": 0}
    transport = GoogleApiSheetsTransport.from_service_account(
        settings.google_service_account_file,
        settings.google_sheets_spreadsheet_id,
    )
    return SheetsSynchronizer(
        sheets_outbox, GoogleSheetsMirror(transport)
    ).sync_pending(limit=limit)


def retry_failed_google_sheets(limit: int = 100) -> dict[str, int]:
    """Explicit admin-only retry of terminally failed outbox events."""
    if sheets_outbox is None:
        return {"sent": 0, "failed": 0, "skipped": 0}
    transport = GoogleApiSheetsTransport.from_service_account(
        settings.google_service_account_file,
        settings.google_sheets_spreadsheet_id,
    )
    return SheetsSynchronizer(
        sheets_outbox, GoogleSheetsMirror(transport)
    ).retry_failed(limit=limit)


@st.cache_data(ttl=5, show_spinner=False)
def probe_mcp(server_url: str) -> tuple[bool, str]:
    async def check() -> dict:
        async with SupportMcpClient(server_url, timeout=1.5) as client:
            return await client.health_check()

    try:
        payload = asyncio.run(check())
        return payload.get("status") == "ok", str(payload.get("service", "MCP"))
    except Exception:
        return False, "Start mcp_server.py first"


mcp_online, mcp_detail = probe_mcp(settings.mcp_server_url)


def load_logs() -> pd.DataFrame:
    rows = ticket_logger.read_all()
    if not rows:
        return pd.DataFrame(columns=TicketLogger.FIELDNAMES)
    frame = pd.DataFrame(rows)
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], errors="coerce", utc=True)
    frame["confidence"] = pd.to_numeric(frame["confidence"], errors="coerce").fillna(0.0)
    return frame


def mask_customer_id(value: object) -> str:
    text = str(value or "anonymous")
    if text.lower() == "anonymous":
        return "Anonymous"
    return text[:2] + "***" + text[-2:] if len(text) > 4 else "***"


def mask_sensitive_text(value: object) -> str:
    text = str(value or "")
    text = re.sub(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", "[EMAIL]", text, flags=re.I)
    text = re.sub(r"(?<!\d)(?:\+?\d[\d ()-]{7,}\d)(?!\d)", "[PHONE]", text)
    text = re.sub(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)", "[PAYMENT_NUMBER]", text)
    return text


def send_human_review(ticket: pd.Series, reply_text: str, final_status: str) -> dict:
    ticket_id = str(ticket["ticket_id"])
    delivery_id = uuid4().hex
    claimed = ticket_logger.claim_review_delivery(
        ticket_id,
        delivery_id=delivery_id,
        review_reply=reply_text,
        final_status=final_status,
    )
    if not claimed:
        return {
            "sent": False,
            "detail": "This ticket is already being sent or has already been reviewed.",
        }
    try:
        mailer = settings.build_mailer()
        result = mailer.send_review_reply(
            ticket_id=ticket_id,
            user_id=str(ticket["user_id"]),
            original_query=str(ticket["query"]),
            reply_text=reply_text,
        )
    except Exception as exc:
        logger.exception("Human review email preparation or delivery failed")
        result = {
            "sent": False,
            "detail": f"Review email failed: {type(exc).__name__}",
        }
    finalized = ticket_logger.finalize_review_delivery(
        ticket_id,
        delivery_id=delivery_id,
        sent=bool(result.get("sent")),
        detail=str(result.get("detail", "unknown")),
    )
    if not finalized:
        return {
            "sent": bool(result.get("sent")),
            "detail": (
                "Email delivery completed, but final ticket-state persistence failed. "
                "The delivery remains claimed to prevent a duplicate send."
            ),
        }
    return result


def chart_or_empty(chart: alt.Chart | None, message: str) -> None:
    if chart is None:
        st.markdown(f'<div class="empty-state">{message}</div>', unsafe_allow_html=True)
    else:
        st.altair_chart(chart, width="stretch")


def run_support_workflow(query: str, user_id: str):
    async def execute():
        model_client = settings.build_model_client()
        try:
            async with SupportMcpClient(
                settings.mcp_server_url, timeout=settings.mcp_timeout_seconds
            ) as mcp_client:
                await mcp_client.health_check()
                workflow = SupportWorkflow(
                    query_agent=CustomerQueryAgent(
                        model_client,
                        product_store,
                        settings.retrieval_min_score,
                        use_local_tools=False,
                    ),
                    knowledge_agent=KnowledgeBaseAgent(
                        model_client,
                        support_store,
                        settings.retrieval_min_score,
                        use_local_tools=False,
                        rerank_enabled=settings.llm_rerank_enabled,
                    ),
                    resolution_agent=ResolutionAgent(model_client),
                    product_store=product_store,
                    support_store=support_store,
                    ticket_logger=ticket_logger,
                    escalation_mailer=settings.build_mailer(),
                    confidence_threshold=settings.confidence_threshold,
                    retrieval_min_score=settings.retrieval_min_score,
                    retrieval_client=mcp_client,
                )
                return await workflow.process(query, user_id)
        finally:
            await model_client.close()

    return asyncio.run(execute())


with st.sidebar:
    st.markdown(
        """<div class="brand"><div class="brand-mark">S</div><div>
        <div class="brand-name">SupportIQ</div><div class="brand-sub">SUPPORT OPERATIONS</div>
        </div></div>""",
        unsafe_allow_html=True,
    )
    page = st.radio(
        "Navigation",
        [
            "Overview",
            "Customer Queries",
            "Log Analytics",
            "Knowledge Coverage",
            "Admin Console",
            "Settings",
        ],
        label_visibility="collapsed",
    )
    st.markdown(
        """<div class="pipeline-title">Agent Pipeline</div>
        <div class="agent-card"><div class="agent-step">1</div><div><div class="agent-name">Customer Query Agent</div><div class="agent-desc">Classifies intent, urgency, sentiment and product scope</div></div></div>
        <div class="agent-card"><div class="agent-step">2</div><div><div class="agent-name">Knowledge Base Agent</div><div class="agent-desc">Retrieves verified company knowledge through MCP</div></div></div>
        <div class="agent-card"><div class="agent-step">3</div><div><div class="agent-name">Resolution Agent</div><div class="agent-desc">Drafts a grounded answer from retrieved evidence</div></div></div>
        <div class="agent-card"><div class="agent-step">4</div><div><div class="agent-name">Log Agent</div><div class="agent-desc">Audits the decision and routes human escalation</div></div></div>""",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.metric("MCP Tool Server", "Online" if mcp_online else "Offline")
    if mcp_online:
        st.success(f"Connected · {mcp_detail}")
    else:
        st.warning(mcp_detail)
    st.caption(safe_mcp_url_for_display(settings.mcp_server_url))
    st.caption(f"Model · {settings.model_name}")
    st.caption(f"Decision gate · {settings.confidence_threshold:.0%}")
    st.caption("Email · " + ("Enabled" if settings.email_enabled else "Disabled"))

logs = load_logs()

if page == "Overview":
    header_left, date_col, category_col, refresh_col = st.columns([5.4, 1.4, 1.5, .7], vertical_alignment="bottom")
    with header_left:
        st.title("Customer Support Analytics")
        st.markdown('<div class="page-subtitle">AI-powered customer support intelligence</div>', unsafe_allow_html=True)
        st.markdown('<div class="eyebrow">Operations intelligence · Live data</div>', unsafe_allow_html=True)
    with date_col:
        date_range = st.selectbox("Date Range", ["All Time", "Last 7 Days", "Last 30 Days", "Last 90 Days"])
    categories = sorted(logs["category"].dropna().unique().tolist()) if not logs.empty else []
    with category_col:
        category_filter = st.selectbox("Category Filter", ["All Categories", *categories])
    with refresh_col:
        if st.button("↻", help="Refresh live data", width="stretch"):
            st.rerun()

    filtered = logs.copy()
    if not filtered.empty and date_range != "All Time":
        days = int(date_range.split()[1])
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
        filtered = filtered[filtered["timestamp_utc"] >= cutoff]
    if category_filter != "All Categories" and not filtered.empty:
        filtered = filtered[filtered["category"] == category_filter]

    st.markdown(
        """<div class="command-bar">
        <div class="command-title"><span class="live-dot"></span>Live operations command center</div>
        <div class="command-meta">Four-agent orchestration · Human approval · Local audit trail</div>
        </div>""",
        unsafe_allow_html=True,
    )

    total_queries = len(filtered)
    resolved_count = int((filtered["status"] == "resolved").sum()) if total_queries else 0
    accuracy = resolved_count / total_queries if total_queries else 0.0
    unique_sources = len({record.get("source") for record in support_store.records if record.get("source")})

    st.markdown('<div class="section-kicker">Operational snapshot</div>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Queries Received", f"{total_queries:,}")
    m2.metric("Auto-resolution Rate", f"{accuracy:.1%}", help="Resolved automatically ÷ total processed queries")
    m3.metric("Total Logs Collected", f"{len(logs):,}")
    m4.metric("Knowledge Vectors", f"{support_store.index.ntotal:,}", delta=f"{unique_sources} sources")

    st.markdown('<div class="section-kicker">Priority action queue</div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.subheader("Human support review")
        st.caption(
            "Review escalated or open tickets, edit the proposed reply, and explicitly approve the email."
        )
        if st.session_state.get("review_notice"):
            st.success(st.session_state.pop("review_notice"))
        review_candidates = (
            logs[logs["status"].isin(["escalated", "open"])].copy()
            if not logs.empty
            else pd.DataFrame()
        )
        if review_candidates.empty:
            st.success("No tickets are waiting for human review.")
        else:
            review_candidates = review_candidates.sort_values("timestamp_utc", ascending=False)
            ticket_options = review_candidates["ticket_id"].astype(str).tolist()
            selected_ticket_id = st.selectbox(
                "Ticket awaiting review",
                ticket_options,
                format_func=lambda ticket_id: (
                    f"{ticket_id} · "
                    f"{review_candidates.loc[review_candidates['ticket_id'] == ticket_id, 'category'].iloc[0].title()} · "
                    f"{review_candidates.loc[review_candidates['ticket_id'] == ticket_id, 'priority'].iloc[0].title()} priority"
                ),
            )
            selected_ticket = review_candidates[
                review_candidates["ticket_id"] == selected_ticket_id
            ].iloc[0]
            tag1, tag2, tag3, tag4 = st.columns(4)
            tag1.metric("Category", str(selected_ticket["category"]).title())
            tag2.metric("Urgency", str(selected_ticket["urgency"]).title())
            tag3.metric("Sentiment", str(selected_ticket["sentiment"]).title())
            tag4.metric("Confidence", f"{float(selected_ticket['confidence']):.0%}")
            st.markdown(
                f"**Customer:** {html.escape(mask_customer_id(selected_ticket['user_id']))}  "
                f"\n**Complaint:** {html.escape(mask_sensitive_text(selected_ticket['query']))}"
            )
            proposed_reply = str(selected_ticket.get("answer", "")).strip() or (
                "Thank you for contacting support. We reviewed your request and will follow up "
                "with the verified next steps."
            )
            reply_to_send = st.text_area(
                "Reply to send",
                value=proposed_reply,
                height=145,
                key=f"review_reply_{selected_ticket_id}",
                help="This AI draft is visible to the human reviewer only and is not sent until approved.",
            )
            action_col, recipient_col = st.columns([1, 1.4], vertical_alignment="bottom")
            with action_col:
                final_status_label = st.selectbox(
                    "Ticket status after approval",
                    ["Resolved", "Open"],
                    key=f"review_status_{selected_ticket_id}",
                )
            with recipient_col:
                st.info(
                    "Escalation mailbox: "
                    + (settings.support_email or "Not configured in .env")
                )
            if st.button(
                "✅ Approve reply and send escalation email",
                type="primary",
                key=f"approve_review_{selected_ticket_id}",
            ):
                if not reply_to_send.strip():
                    st.error("Enter a reply before approval.")
                else:
                    with st.spinner("Sending approved review email…"):
                        email_result = send_human_review(
                            selected_ticket,
                            reply_to_send,
                            final_status_label.lower(),
                        )
                    if email_result.get("sent"):
                        st.session_state.review_notice = str(
                            email_result.get("detail", "Email sent.")
                        )
                        st.rerun()
                    else:
                        st.error(str(email_result.get("detail", "Email was not sent.")))

    st.markdown('<div class="section-kicker">Ticket distribution</div>', unsafe_allow_html=True)
    chart_left, chart_right = st.columns(2, gap="large")
    with chart_left:
        with st.container(border=True):
            st.subheader("Tickets per category")
            st.caption("What customers contact support about")
            if filtered.empty:
                chart_or_empty(None, "No category data available.")
            else:
                category_counts = (
                    filtered.groupby("category").size().reset_index(name="tickets")
                )
                chart_or_empty(
                    alt.Chart(category_counts).mark_bar(
                        color="#10b981", cornerRadiusTopLeft=5, cornerRadiusTopRight=5
                    ).encode(
                        x=alt.X("category:N", sort="-y", title=None),
                        y=alt.Y("tickets:Q", title="Tickets", axis=alt.Axis(tickMinStep=1)),
                        tooltip=["category:N", "tickets:Q"],
                    ).properties(height=245),
                    "",
                )
    with chart_right:
        with st.container(border=True):
            st.subheader("Ticket status")
            st.caption("Open, automatically resolved, and escalated tickets")
            status_order = ["open", "resolved", "escalated"]
            counts = (
                filtered.groupby("status").size().to_dict() if not filtered.empty else {}
            )
            status_counts = pd.DataFrame(
                [
                    {"status": status.title(), "tickets": int(counts.get(status, 0))}
                    for status in status_order
                ]
            )
            chart_or_empty(
                alt.Chart(status_counts).mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5).encode(
                    x=alt.X("status:N", sort=["Open", "Resolved", "Escalated"], title=None),
                    y=alt.Y("tickets:Q", title="Tickets", axis=alt.Axis(tickMinStep=1)),
                    color=alt.Color(
                        "status:N",
                        scale=alt.Scale(
                            domain=["Open", "Resolved", "Escalated"],
                            range=["#f3a51b", "#20a35a", "#dc4545"],
                        ),
                        legend=None,
                    ),
                    tooltip=["status:N", "tickets:Q"],
                ).properties(height=245),
                "",
            )

    st.markdown('<div class="section-kicker">Audit trail</div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.subheader("Ticket log")
        st.caption("Every customer request and agent decision is recorded in the local CSV audit log.")
        if filtered.empty:
            st.info("No tickets have been recorded yet.")
        else:
            ticket_view = filtered.copy()
            ticket_view["customer"] = ticket_view["user_id"].map(mask_customer_id)
            ticket_view["complaint"] = ticket_view["query"].map(mask_sensitive_text)
            visible_columns = [
                "ticket_id",
                "timestamp_utc",
                "customer",
                "complaint",
                "category",
                "urgency",
                "sentiment",
                "priority",
                "confidence",
                "status",
            ]
            st.dataframe(
                ticket_view[visible_columns].sort_values("timestamp_utc", ascending=False),
                hide_index=True,
                width="stretch",
                height=330,
            )
            st.download_button(
                "Download masked tickets.csv",
                ticket_view[visible_columns].to_csv(index=False).encode("utf-8"),
                "customer_support_tickets_masked.csv",
                "text/csv",
            )


elif page == "Customer Queries":
    st.title("Customer Queries")
    st.markdown('<div class="page-subtitle">Run the complete four-agent AutoGen support workflow</div>', unsafe_allow_html=True)
    form_col, gate_col = st.columns([1.5, .75], gap="large")
    with form_col:
        with st.container(border=True):
            st.subheader("New support request")
            with st.form("support_request"):
                user_id = st.text_input("Customer ID", placeholder="customer-001")
                query = st.text_area("Customer query", height=160, placeholder="Describe the issue, what happened, and the expected outcome.")
                submitted = st.form_submit_button("Run four-agent workflow", width="stretch")
    with gate_col:
        with st.container(border=True):
            st.subheader("Automatic decision gates")
            st.markdown(f"- Confidence **≥ {settings.confidence_threshold:.0%}**\n- Grounded in retrieved evidence\n- Inside product/service scope\n- High-priority cases escalate\n- Only verified resolutions enter memory")

    if submitted:
        if product_store.index.ntotal == 0:
            st.error("Ingest at least one product/support PDF under Knowledge Coverage first.")
        elif not query.strip():
            st.error("Enter a customer query.")
        else:
            try:
                with st.spinner("Agents are classifying, retrieving, resolving, and logging…"):
                    st.session_state.last_result = run_support_workflow(query, user_id)
            except Exception as exc:
                logger.exception("Support workflow failed")
                if isinstance(exc, ConnectionError):
                    st.error("The MCP tool server is unavailable. Start mcp_server.py and retry.")
                else:
                    st.error("The workflow could not complete. Review the application logs and retry.")

    result = st.session_state.get("last_result")
    if result:
        css_class = "status-resolved" if result.status == "resolved" else "status-escalated"
        st.markdown(f'<div class="{css_class}"><b>{result.status.upper()}</b> · Ticket {result.ticket_id}</div>', unsafe_allow_html=True)
        st.subheader("Final response")
        st.write(result.customer_message)
        c1, c2, c3 = st.columns(3)
        c1.metric("Confidence", f"{result.resolution.confidence:.0%}")
        c2.metric("Priority", result.classification.priority.title())
        c3.metric("Sentiment", result.classification.sentiment.title())
        if result.escalation_reason:
            st.warning(result.escalation_reason)
            st.caption(result.email_status)
        st.caption("Sources: " + (", ".join(result.resolution.source_filenames) or "None"))
        st.subheader("Agent timeline")
        for stage in result.stages:
            with st.expander(f"✓ {stage['agent']}"):
                st.json(stage["output"])
        with st.expander("Retrieved evidence"):
            for passage in result.passages:
                st.markdown(f"**{passage.get('source')} · page {passage.get('page')} · score {passage.get('score')}**")
                st.write(passage.get("text"))

elif page == "Log Analytics":
    st.title("Log Analytics")
    st.markdown('<div class="page-subtitle">Auditable tickets, agent decisions, and escalation outcomes</div>', unsafe_allow_html=True)
    a, b, c, d = st.columns(4)
    a.metric("Total Logs Collected", f"{len(logs):,}")
    b.metric("Open", f"{int((logs['status'] == 'open').sum()) if not logs.empty else 0:,}")
    c.metric("Resolved", f"{int((logs['status'] == 'resolved').sum()) if not logs.empty else 0:,}")
    d.metric("Escalated", f"{int((logs['status'] == 'escalated').sum()) if not logs.empty else 0:,}")
    if logs.empty:
        st.info("No tickets logged yet.")
    else:
        chart_left, chart_right = st.columns(2, gap="large")
        with chart_left:
            with st.container(border=True):
                st.subheader("Ticket volume trend")
                volume = (
                    logs.groupby(logs["timestamp_utc"].dt.floor("D").rename("date"))
                    .size()
                    .reset_index(name="tickets")
                )
                st.altair_chart(
                    alt.Chart(volume).mark_area(
                        line={"color": "#3478e5", "strokeWidth": 2.5},
                        color=alt.Gradient(
                            gradient="linear",
                            stops=[
                                alt.GradientStop(color="#3478e5", offset=0),
                                alt.GradientStop(color="#e9f1ff", offset=1),
                            ],
                            x1=1,
                            x2=1,
                            y1=0,
                            y2=1,
                        ),
                    ).encode(
                        x=alt.X("date:T", title=None),
                        y=alt.Y("tickets:Q", title="Tickets"),
                        tooltip=[alt.Tooltip("date:T"), "tickets:Q"],
                    ).properties(height=230),
                    width="stretch",
                )
        with chart_right:
            with st.container(border=True):
                st.subheader("Demand by category")
                categories = logs.groupby("category").size().reset_index(name="tickets")
                st.altair_chart(
                    alt.Chart(categories).mark_bar(color="#7b61e8", cornerRadiusEnd=4).encode(
                        y=alt.Y("category:N", sort="-x", title=None),
                        x=alt.X("tickets:Q", title="Tickets"),
                        tooltip=["category:N", "tickets:Q"],
                    ).properties(height=230),
                    width="stretch",
                )

        insight_left, insight_right = st.columns(2, gap="large")
        with insight_left:
            with st.container(border=True):
                st.subheader("Priority and sentiment mix")
                mix = logs.groupby(["priority", "sentiment"]).size().reset_index(name="tickets")
                st.altair_chart(
                    alt.Chart(mix).mark_rect(cornerRadius=4).encode(
                        x=alt.X("sentiment:N", title="Sentiment"),
                        y=alt.Y("priority:N", title="Priority"),
                        color=alt.Color("tickets:Q", scale=alt.Scale(scheme="blues")),
                        tooltip=["priority:N", "sentiment:N", "tickets:Q"],
                    ).properties(height=210),
                    width="stretch",
                )
        with insight_right:
            with st.container(border=True):
                st.subheader("Resolution confidence")
                confidence = logs.copy()
                confidence["confidence"] = pd.to_numeric(confidence["confidence"], errors="coerce")
                confidence = confidence.dropna(subset=["confidence"])
                if confidence.empty:
                    st.info("No confidence values available.")
                else:
                    st.altair_chart(
                        alt.Chart(confidence).mark_bar(color="#20ad87").encode(
                            x=alt.X("confidence:Q", bin=alt.Bin(maxbins=10), title="Confidence"),
                            y=alt.Y("count():Q", title="Tickets"),
                            tooltip=[alt.Tooltip("count():Q", title="Tickets")],
                        ).properties(height=210),
                        width="stretch",
                    )

        with st.container(border=True):
            st.subheader("Ticket audit trail")
            audit_view = logs.copy()
            audit_view["user_id"] = audit_view["user_id"].map(mask_customer_id)
            for column in ["query", "answer", "review_reply"]:
                if column in audit_view.columns:
                    audit_view[column] = audit_view[column].map(mask_sensitive_text)
            st.dataframe(
                audit_view.sort_values("timestamp_utc", ascending=False),
                hide_index=True,
                width="stretch",
            )
            masked_csv = audit_view.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Export masked CSV",
                masked_csv,
                "customer_support_tickets_masked.csv",
                "text/csv",
            )

elif page == "Knowledge Coverage":
    st.title("Knowledge Coverage")
    st.markdown('<div class="page-subtitle">Manage product scope, policies, FAQs, catalogs, and resolved-case memory</div>', unsafe_allow_html=True)
    upload_col, summary_col = st.columns([1.25, .75], gap="large")
    with upload_col:
        with st.container(border=True):
            st.subheader("Ingest PDF knowledge")
            uploads = st.file_uploader("Upload text-based PDFs", type=["pdf"], accept_multiple_files=True)
            target_label = st.selectbox(
                "Index destination",
                ["Both agent stores", "Product scope only", "Support knowledge only"],
                help="Use product scope for manuals/catalogs and support knowledge for policies/FAQs. Choose both when a document serves both agents.",
            )
            target = {
                "Both agent stores": "both",
                "Product scope only": "product",
                "Support knowledge only": "support",
            }[target_label]
            if st.button("Ingest PDFs", type="primary", width="stretch", disabled=not uploads):
                reports = []
                for upload in uploads or []:
                    if upload.size > 20 * 1024 * 1024:
                        st.error(f"{upload.name}: exceeds 20 MB")
                        continue
                    try:
                        reports.append(
                            ingest_pdf(
                                upload.getvalue(),
                                filename=upload.name,
                                product_store=product_store,
                                support_store=support_store,
                                target=target,
                                graph=knowledge_graph,
                            )
                        )
                    except Exception:
                        logger.exception("PDF ingestion failed for %s", upload.name)
                        st.error(f"{upload.name}: ingestion failed; review the application logs.")
                if reports:
                    st.success(f"Processed {len(reports)} PDF(s).")
                    st.json(reports)
    with summary_col:
        p1, p2 = st.columns(2)
        p1.metric("Product Vectors", f"{product_store.index.ntotal:,}")
        p2.metric("Support Vectors", f"{support_store.index.ntotal:,}")
        with st.container(border=True):
            st.subheader("Indexed sources")
            sources = sorted({str(record.get("source")) for record in support_store.records if record.get("source")})
            if not sources:
                st.caption("No documents indexed.")
            else:
                st.markdown(
                    "".join(
                        f'<span class="source-pill">{html.escape(source)}</span>'
                        for source in sources
                    ),
                    unsafe_allow_html=True,
                )
    with st.container(border=True):
        st.subheader("Memory composition")
        if not support_store.records:
            st.info("No vectors available.")
        else:
            composition = pd.DataFrame(support_store.records)
            summary = composition.groupby(["source", "kind"]).size().reset_index(name="vectors").sort_values("vectors", ascending=False)
            st.dataframe(summary, hide_index=True, width="stretch")

elif page == "Admin Console":
    st.title("Admin Console")
    st.markdown(
        '<div class="page-subtitle">Local administrator surface for operational health, retrieval provenance, and external synchronization</div>',
        unsafe_allow_html=True,
    )
    st.warning(
        "This console has no multi-user authentication. Keep it localhost only; do not expose it to a LAN or the internet."
    )

    st.subheader("Ticket operations")
    ticket_total = len(logs)
    pending_review = (
        int(logs["status"].isin(["open", "escalated"]).sum()) if ticket_total else 0
    )
    email_failures = (
        int(logs["escalation_delivery_state"].eq("failed").sum())
        if ticket_total and "escalation_delivery_state" in logs.columns
        else 0
    )
    outbox_records = sheets_outbox.read_all() if sheets_outbox is not None else []
    sync_pending_count = sum(item.get("state") == "pending" for item in outbox_records)
    op1, op2, op3, op4 = st.columns(4)
    op1.metric("Tickets", f"{ticket_total:,}")
    op2.metric("Awaiting review", f"{pending_review:,}")
    op3.metric("Email failures", f"{email_failures:,}")
    op4.metric("Sheets pending", f"{sync_pending_count:,}")
    if logs.empty:
        st.info("No ticket operations are available yet.")
    else:
        operations = logs.copy()
        operations["customer"] = operations["user_id"].map(mask_customer_id)
        operation_columns = [
            column
            for column in (
                "ticket_id",
                "timestamp_utc",
                "customer",
                "category",
                "priority",
                "status",
                "escalation_delivery_state",
                "review_email_status",
                "total_latency_ms",
            )
            if column in operations.columns
        ]
        st.dataframe(
            operations[operation_columns].sort_values("timestamp_utc", ascending=False),
            hide_index=True,
            width="stretch",
            height=280,
        )

    st.subheader("Latency and status")
    latency_col, status_col = st.columns(2, gap="large")
    with latency_col:
        if logs.empty or "total_latency_ms" not in logs.columns:
            st.info("Latency appears after newly processed tickets complete.")
        else:
            latency = logs[["ticket_id", "total_latency_ms"]].copy()
            latency["total_latency_ms"] = pd.to_numeric(
                latency["total_latency_ms"], errors="coerce"
            )
            latency = latency.dropna(subset=["total_latency_ms"]).tail(30)
            if latency.empty:
                st.info("No completed latency measurements yet.")
            else:
                st.altair_chart(
                    alt.Chart(latency).mark_bar(color="#10b981", cornerRadiusEnd=4).encode(
                        y=alt.Y("ticket_id:N", sort=None, title=None),
                        x=alt.X("total_latency_ms:Q", title="Total latency (ms)"),
                        tooltip=["ticket_id:N", "total_latency_ms:Q"],
                    ).properties(height=260),
                    width="stretch",
                )
    with status_col:
        if logs.empty:
            st.info("No ticket state available.")
        else:
            state_counts = logs.groupby("status").size().reset_index(name="tickets")
            st.altair_chart(
                alt.Chart(state_counts).mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5).encode(
                    x=alt.X("status:N", title=None),
                    y=alt.Y("tickets:Q", title="Tickets"),
                    color=alt.Color(
                        "status:N",
                        scale=alt.Scale(
                            domain=["open", "resolved", "escalated"],
                            range=["#f59e0b", "#10b981", "#dc4545"],
                        ),
                        legend=None,
                    ),
                    tooltip=["status:N", "tickets:Q"],
                ).properties(height=260),
                width="stretch",
            )

    if not logs.empty and "stage_latencies_ms" in logs.columns:
        with st.expander("Per-stage latency details"):
            latency_ticket_ids = logs["ticket_id"].astype(str).tolist()
            selected_latency_ticket = st.selectbox(
                "Ticket",
                list(reversed(latency_ticket_ids)),
                key="admin_latency_ticket",
            )
            raw_stages = logs.loc[
                logs["ticket_id"].astype(str) == selected_latency_ticket,
                "stage_latencies_ms",
            ].iloc[0]
            try:
                parsed_stages = json.loads(str(raw_stages)) if str(raw_stages).strip() else {}
            except json.JSONDecodeError:
                parsed_stages = {}
            stage_rows = [
                {"stage": str(stage).replace("_", " ").title(), "latency_ms": float(value)}
                for stage, value in parsed_stages.items()
                if isinstance(value, (int, float))
            ]
            if stage_rows:
                st.altair_chart(
                    alt.Chart(pd.DataFrame(stage_rows)).mark_bar(
                        color="#2563eb", cornerRadiusEnd=4
                    ).encode(
                        y=alt.Y("stage:N", sort="-x", title=None),
                        x=alt.X("latency_ms:Q", title="Latency (ms)"),
                        tooltip=["stage:N", "latency_ms:Q"],
                    ).properties(height=220),
                    width="stretch",
                )
            else:
                st.caption("No stage-level measurement is available for this ticket.")

    st.subheader("Retrieval inspector")
    retrieval_left, retrieval_right = st.columns([0.8, 1.8], gap="large")
    with retrieval_left:
        inspector_store_name = st.selectbox(
            "Evidence store",
            ["Support knowledge", "Product scope"],
            key="admin_retrieval_store",
        )
        inspector_query = st.text_input(
            "Inspector query",
            placeholder="refund delay, login error, product name…",
            key="admin_retrieval_query",
        )
        inspector_limit = st.slider("Maximum results", 1, 12, 6, key="admin_retrieval_limit")
    with retrieval_right:
        active_store = support_store if inspector_store_name == "Support knowledge" else product_store
        st.caption(
            f"{active_store.index.ntotal:,} indexed chunks · component ranks are shown instead of raw vector arrays"
        )
        if inspector_query.strip():
            evidence_hits = active_store.search(
                inspector_query,
                top_k=inspector_limit,
                min_score=0.0,
            )
            if not evidence_hits:
                st.info("No evidence matched this query.")
            else:
                evidence_view = pd.DataFrame(evidence_hits)
                evidence_view["preview"] = evidence_view["text"].map(
                    lambda value: mask_sensitive_text(value)[:180]
                )
                evidence_columns = [
                    column
                    for column in (
                        "chunk_id",
                        "source",
                        "page",
                        "kind",
                        "vector_rank",
                        "vector_score",
                        "bm25_rank",
                        "bm25_score",
                        "fusion_score",
                        "score",
                        "preview",
                    )
                    if column in evidence_view.columns
                ]
                st.dataframe(evidence_view[evidence_columns], hide_index=True, width="stretch")
        else:
            st.info("Enter a query to inspect hybrid BM25, FAISS, and fusion ranking.")

    st.subheader("Knowledge graph viewer")
    graph_stats = knowledge_graph.stats()
    g1, g2, g3, g4 = st.columns(4)
    g1.metric("Graph nodes", f"{graph_stats['nodes']:,}")
    g2.metric("Graph edges", f"{graph_stats['edges']:,}")
    g3.metric("Documents", f"{graph_stats['documents']:,}")
    g4.metric("Chunks", f"{graph_stats['chunks']:,}")
    graph_query = st.text_input(
        "Graph evidence query",
        placeholder="Search redacted provenance previews",
        key="admin_graph_query",
    )
    if graph_query.strip():
        graph_result = knowledge_graph.search(graph_query, limit=10, max_nodes=50)
        if not graph_result["hits"]:
            st.info("No graph evidence matched this query.")
        else:
            st.dataframe(pd.DataFrame(graph_result["hits"]), hide_index=True, width="stretch")
            with st.expander("Bounded evidence subgraph"):
                node_tab, edge_tab = st.tabs(["Nodes", "Edges"])
                with node_tab:
                    st.dataframe(pd.DataFrame(graph_result["nodes"]), hide_index=True, width="stretch")
                with edge_tab:
                    st.dataframe(pd.DataFrame(graph_result["edges"]), hide_index=True, width="stretch")
    else:
        st.info("Enter a query to inspect source, page, chunk, and document-kind relationships.")

    st.subheader("Google Sheets synchronization")
    if not settings.google_sheets_enabled:
        st.info("Google Sheets mirroring is disabled. Local CSV remains the authoritative ticket log.")
    elif not settings.google_sheets_configured or sheets_outbox is None:
        st.warning("Google Sheets is enabled but its service-account file or spreadsheet ID is missing.")
    else:
        sync_counts = {
            state: sum(item.get("state") == state for item in outbox_records)
            for state in ("pending", "sent", "failed")
        }
        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Pending", f"{sync_counts['pending']:,}")
        sc2.metric("Sent", f"{sync_counts['sent']:,}")
        sc3.metric("Failed", f"{sync_counts['failed']:,}")
        acknowledge_sync = st.checkbox(
            "I approve sending the masked operational fields to the configured Google Sheet.",
            key="admin_sheets_approval",
        )
        if st.button(
            "Synchronize pending events",
            type="primary",
            disabled=not acknowledge_sync or sync_counts["pending"] == 0,
            key="admin_sheets_sync",
        ):
            try:
                with st.spinner("Synchronizing masked ticket events…"):
                    sync_result = sync_google_sheets(limit=100)
                st.success(
                    f"Synchronization completed: {sync_result['sent']} sent, "
                    f"{sync_result['failed']} failed."
                )
                st.rerun()
            except Exception:
                logger.exception("Google Sheets synchronization failed")
                st.error("Synchronization failed. Local tickets are safe; review the application logs and retry.")
        if st.button(
            "Retry failed events",
            disabled=not acknowledge_sync or sync_counts["failed"] == 0,
            key="admin_sheets_retry_failed",
        ):
            try:
                with st.spinner("Retrying failed masked ticket events…"):
                    retry_result = retry_failed_google_sheets(limit=100)
                st.success(
                    f"Retry completed: {retry_result['sent']} sent, "
                    f"{retry_result['failed']} failed."
                )
                st.rerun()
            except Exception:
                logger.exception("Google Sheets failed-event retry failed")
                st.error("Retry failed. Local tickets are safe; review the application logs.")

else:
    st.title("Settings")
    st.markdown('<div class="page-subtitle">Runtime configuration — secret values are never displayed</div>', unsafe_allow_html=True)
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Model", settings.model_name)
    s2.metric("Confidence Gate", f"{settings.confidence_threshold:.0%}")
    s3.metric("MCP Tool Server", "Online" if mcp_online else "Offline")
    s4.metric("Escalation Email", "Enabled" if settings.email_enabled else "Disabled")
    with st.container(border=True):
        st.subheader("MCP integration")
        st.code(safe_mcp_url_for_display(settings.mcp_server_url))
        if mcp_online:
            st.success("The app can discover and call all registered support tools.")
        else:
            st.warning("Start the MCP process before submitting customer queries: python mcp_server.py")
    with st.container(border=True):
        st.subheader("Storage")
        st.code(str(Path(settings.data_dir).resolve()))
        st.caption("Configure values in .env. Keep API and SMTP credentials out of source control.")
