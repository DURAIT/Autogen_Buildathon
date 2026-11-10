from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
import time

from support_system.models import Classification, Resolution
from support_system.sheets_mirror import (
    CURRENT_COLUMNS,
    GoogleApiSheetsTransport,
    GoogleSheetsMirror,
    SheetsSynchronizer,
    TicketOutbox,
)
from support_system.config import Settings
from support_system.storage import TicketLogger


def ticket_row(**overrides):
    row = {
        "ticket_id": "T-1",
        "timestamp_utc": "2026-01-01T00:00:00Z",
        "user_id": "customer@example.com",
        "query": "card 4111111111111111",
        "category": "=refund",
        "priority": "high",
        "status": "escalated",
        "confidence": "0.5",
        "escalation_delivery_state": "sending",
        "reviewed_at_utc": "",
        "answer": "secret answer",
        "review_reply": "secret reply",
        "escalation_delivery_id": "secret-id",
    }
    row.update(overrides)
    return row


def test_outbox_persists_only_approved_formula_neutral_payload_and_deduplicates(tmp_path: Path):
    outbox = TicketOutbox(tmp_path / "sheets_outbox.jsonl")

    first = outbox.enqueue(ticket_row())
    second = outbox.enqueue(ticket_row())

    assert first["event_id"] == second["event_id"]
    assert len(outbox.read_all()) == 1
    stored = json.loads((tmp_path / "sheets_outbox.jsonl").read_text().splitlines()[0])
    assert stored["state"] == "pending"
    assert stored["attempts"] == 0
    assert stored["snapshot_hash"]
    assert stored["snapshot_version"]
    assert stored["payload"]["category"] == "other"
    assert stored["payload"]["masked_customer_id"] == "cu***om"
    forbidden = {"query", "answer", "review_reply", "user_id", "email", "phone", "payment", "escalation_delivery_id"}
    assert forbidden.isdisjoint(stored["payload"])
    serialized = json.dumps(stored)
    assert "customer@example.com" not in serialized
    assert "4111111111111111" not in serialized
    assert "secret answer" not in serialized


def test_outbound_payload_redacts_every_string_field(tmp_path: Path):
    outbox = TicketOutbox(tmp_path / "sheets_outbox.jsonl")
    record = outbox.enqueue(
        ticket_row(
            category="refund customer@example.com",
            priority="Call +1 212 555 0100",
            status="card 4111111111111111",
            stage_latencies_ms="token=supersecret",
        )
    )

    serialized = json.dumps(record["payload"])
    assert record["payload"]["category"] == "other"
    assert "customer@example.com" not in serialized
    assert "212 555 0100" not in serialized
    assert "4111111111111111" not in serialized
    assert "supersecret" not in serialized


def test_ticket_logger_enqueues_snapshot_after_successful_append_without_failing_local_write(tmp_path: Path):
    snapshots = []
    logger = TicketLogger(tmp_path / "tickets.csv", event_sink=lambda row: snapshots.append(row))
    classification = Classification(category="billing", urgency="high", sentiment="angry", priority="high", summary="x", is_product_related=True, rationale="x")
    resolution = Resolution(answer="private", confidence=.4, source_filenames=[], grounded=False)

    logger.append(ticket_id="T-2", user_id="private@example.com", query="private", classification=classification, resolution=resolution, status="escalated", escalation_reason="low")

    assert logger.read_all()[0]["ticket_id"] == "T-2"
    assert snapshots[-1]["ticket_id"] == "T-2"

    failing = TicketLogger(tmp_path / "other.csv", event_sink=lambda row: (_ for _ in ()).throw(RuntimeError("outage")))
    failing.append(ticket_id="T-3", user_id="u", query="q", classification=classification, resolution=resolution, status="resolved", escalation_reason="")
    assert failing.read_all()[0]["ticket_id"] == "T-3"


def test_ticket_logger_emits_each_successful_state_mutation(tmp_path: Path):
    snapshots = []
    logger = TicketLogger(tmp_path / "tickets.csv", event_sink=snapshots.append)
    classification = Classification(category="billing", urgency="high", sentiment="angry", priority="high", summary="x", is_product_related=True, rationale="x")
    resolution = Resolution(answer="private", confidence=.4, source_filenames=[], grounded=False)
    logger.append(ticket_id="T-1", user_id="u", query="q", classification=classification, resolution=resolution, status="escalated", escalation_reason="low", escalation_delivery_id="D")
    assert logger.claim_review_delivery("T-1", delivery_id="R", review_reply="reply", final_status="resolved")
    assert logger.finalize_review_delivery("T-1", delivery_id="R", sent=False, detail="failed")
    assert logger.claim_escalation_delivery("T-1", "D")
    assert logger.finalize_escalation_delivery("T-1", sent=False, detail="failed")
    assert logger.mark_escalation_duplicate("T-1")
    assert logger.update_ticket("T-1", status="open", review_reply="reply", review_email_status="saved")
    assert len(snapshots) == 7
    assert snapshots[-1]["status"] == "open"


class FakeTransport:
    def __init__(self, failures=0):
        self.failures = failures
        self.events = []
        self.current = {}

    def mirror(self, event):
        if self.failures:
            self.failures -= 1
            raise OSError("sheets unavailable")
        if event["event_id"] not in {item["event_id"] for item in self.events}:
            self.events.append(event)
        self.current[event["ticket_id"]] = event["payload"]


def test_synchronizer_marks_sent_and_is_idempotent(tmp_path: Path):
    outbox = TicketOutbox(tmp_path / "outbox.jsonl")
    event = outbox.enqueue(ticket_row())
    transport = FakeTransport()
    synchronizer = SheetsSynchronizer(outbox, GoogleSheetsMirror(transport), max_attempts=3, sleep=lambda _: None)

    result = synchronizer.sync_pending(limit=10)
    again = synchronizer.sync_pending(limit=10)

    assert result == {"sent": 1, "failed": 0, "skipped": 0}
    assert again == {"sent": 0, "failed": 0, "skipped": 0}
    assert transport.events[0]["event_id"] == event["event_id"]
    assert outbox.read_all()[0]["state"] == "sent"


def test_synchronizers_atomically_claim_event_before_network_call(tmp_path: Path):
    path = tmp_path / "outbox.jsonl"
    TicketOutbox(path).enqueue(ticket_row())

    class SlowTransport(FakeTransport):
        def __init__(self):
            super().__init__()
            self.call_count = 0
            self.lock = Lock()

        def mirror(self, event):
            with self.lock:
                self.call_count += 1
            time.sleep(0.05)
            super().mirror(event)

    transport = SlowTransport()
    synchronizers = [
        SheetsSynchronizer(TicketOutbox(path), GoogleSheetsMirror(transport), sleep=lambda _: None)
        for _ in range(2)
    ]

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda sync: sync.sync_pending(), synchronizers))

    assert transport.call_count == 1
    assert sum(result["sent"] for result in results) == 1
    assert TicketOutbox(path).read_all()[0]["state"] == "sent"


def test_distinct_events_for_same_ticket_cannot_be_claimed_concurrently(tmp_path: Path):
    path = tmp_path / "outbox.jsonl"
    first_outbox = TicketOutbox(path)
    first_outbox.enqueue(ticket_row(status="open"))
    first_outbox.enqueue(ticket_row(status="resolved"))

    first_claim = first_outbox.claim_pending(1, lease_seconds=60)
    second_claim = TicketOutbox(path).claim_pending(1, lease_seconds=60)

    assert len(first_claim) == 1
    assert second_claim == []


def test_claim_skips_busy_ticket_and_preserves_concurrency_for_other_tickets(tmp_path: Path):
    path = tmp_path / "outbox.jsonl"
    first_outbox = TicketOutbox(path)
    first_outbox.enqueue(ticket_row(ticket_id="T-1", status="open"))
    first_outbox.enqueue(ticket_row(ticket_id="T-1", status="resolved"))
    first_outbox.enqueue(ticket_row(ticket_id="T-2", status="open"))

    first_claim = first_outbox.claim_pending(1, lease_seconds=60)
    second_claim = TicketOutbox(path).claim_pending(1, lease_seconds=60)

    assert first_claim[0]["ticket_id"] == "T-1"
    assert second_claim[0]["ticket_id"] == "T-2"


def test_synchronizer_does_not_report_sent_after_its_lease_expires(tmp_path: Path):
    outbox = TicketOutbox(tmp_path / "outbox.jsonl")
    outbox.enqueue(ticket_row())

    class SlowTransport(FakeTransport):
        def mirror(self, event):
            time.sleep(0.01)
            super().mirror(event)

    synchronizer = SheetsSynchronizer(
        outbox,
        GoogleSheetsMirror(SlowTransport()),
        lease_seconds=0.001,
        sleep=lambda _: None,
    )

    result = synchronizer.sync_pending()

    assert result == {"sent": 0, "failed": 0, "skipped": 1}
    assert outbox.read_all()[0]["state"] == "claimed"


def test_expired_claim_cannot_finalize_and_is_recovered(tmp_path: Path):
    outbox = TicketOutbox(tmp_path / "outbox.jsonl")
    event = outbox.enqueue(ticket_row())
    first = outbox.claim_pending(1, lease_seconds=0.001)[0]
    time.sleep(0.01)

    assert outbox.record_attempt(event["event_id"], first["claim_token"], sent=True) is False
    second = outbox.claim_pending(1, lease_seconds=60)[0]
    assert second["claim_token"] != first["claim_token"]
    assert outbox.record_attempt(event["event_id"], second["claim_token"], sent=True) is True
    assert outbox.read_all()[0]["state"] == "sent"


def test_synchronizer_bounds_retries_and_records_failure(tmp_path: Path):
    outbox = TicketOutbox(tmp_path / "outbox.jsonl")
    outbox.enqueue(ticket_row())
    transport = FakeTransport(failures=5)
    synchronizer = SheetsSynchronizer(outbox, GoogleSheetsMirror(transport), max_attempts=2, sleep=lambda _: None)

    result = synchronizer.sync_pending()
    record = outbox.read_all()[0]

    assert result == {"sent": 0, "failed": 1, "skipped": 0}
    assert record["state"] == "failed"
    assert record["attempts"] == 2
    assert "OSError" in record["error"]


def test_retry_failed_requires_explicit_operation_and_resends(tmp_path: Path):
    outbox = TicketOutbox(tmp_path / "outbox.jsonl")
    outbox.enqueue(ticket_row())
    transport = FakeTransport(failures=1)
    synchronizer = SheetsSynchronizer(
        outbox, GoogleSheetsMirror(transport), max_attempts=1, sleep=lambda _: None
    )
    assert synchronizer.sync_pending()["failed"] == 1
    assert synchronizer.sync_pending()["sent"] == 0

    result = synchronizer.retry_failed(limit=1)

    assert result == {"sent": 1, "failed": 0, "skipped": 0}
    assert outbox.read_all()[0]["state"] == "sent"


def test_retry_failed_targets_selected_event_instead_of_older_pending_event(tmp_path: Path):
    path = tmp_path / "outbox.jsonl"
    outbox = TicketOutbox(path)
    unrelated = outbox.enqueue(ticket_row(ticket_id="T-OLD", status="open"))
    failed = outbox.enqueue(ticket_row(ticket_id="T-FAILED", status="open"))
    claims = outbox.claim_pending(2, lease_seconds=60)
    failed_claim = next(event for event in claims if event["event_id"] == failed["event_id"])
    assert outbox.mark_failed(failed["event_id"], failed_claim["claim_token"], "outage")
    records = outbox.read_all()
    next(record for record in records if record["event_id"] == unrelated["event_id"])[
        "lease_expires_at"
    ] = 0.0
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    assert outbox.claim_pending(0) == []
    assert {record["event_id"]: record["state"] for record in outbox.read_all()} == {
        unrelated["event_id"]: "pending",
        failed["event_id"]: "failed",
    }
    transport = FakeTransport()
    synchronizer = SheetsSynchronizer(
        outbox, GoogleSheetsMirror(transport), max_attempts=1, sleep=lambda _: None
    )

    result = synchronizer.retry_failed(limit=1)

    states = {record["event_id"]: record["state"] for record in outbox.read_all()}
    assert result == {"sent": 1, "failed": 0, "skipped": 0}
    assert [event["event_id"] for event in transport.events] == [failed["event_id"]]
    assert states[failed["event_id"]] == "sent"
    assert states[unrelated["event_id"]] == "pending"


def test_google_transport_does_not_overwrite_newer_current_ticket_snapshot(tmp_path: Path):
    event = TicketOutbox(tmp_path / "outbox.jsonl").enqueue(ticket_row())
    event["snapshot_version"] = 100
    current_row = [event["payload"].get(column, "") for column in CURRENT_COLUMNS]
    current_row[CURRENT_COLUMNS.index("ticket_id")] = event["ticket_id"]
    current_row[CURRENT_COLUMNS.index("snapshot_version")] = 200

    class Request:
        def __init__(self, response=None):
            self.response = response or {}

        def execute(self):
            return self.response

    class Values:
        def __init__(self):
            self.updates = []
            self.appends = []

        def get(self, **kwargs):
            range_name = kwargs["range"]
            if "Ticket Events" in range_name:
                return Request({"values": [["event_id"]]})
            if range_name.endswith("!A:A"):
                return Request({"values": [["ticket_id"], [event["ticket_id"]]]})
            return Request({"values": [list(CURRENT_COLUMNS), current_row]})

        def append(self, **kwargs):
            self.appends.append(kwargs)
            return Request()

        def update(self, **kwargs):
            self.updates.append(kwargs)
            return Request()

    values = Values()

    class Service:
        def spreadsheets(self):
            return self

        def values(self):
            return values

    GoogleApiSheetsTransport(Service(), "sheet-id").mirror(event)

    assert values.updates == []
    assert len(values.appends) == 1  # Event history remains append-only and idempotent.


def test_google_sheets_config_is_disabled_by_default_and_requires_complete_config(monkeypatch):
    for key in ("ENABLE_GOOGLE_SHEETS_SYNC", "GOOGLE_SERVICE_ACCOUNT_FILE", "GOOGLE_SHEETS_SPREADSHEET_ID"):
        monkeypatch.delenv(key, raising=False)
    settings = Settings.from_env()
    assert settings.retrieval_min_score == 0.20
    assert settings.google_sheets_enabled is False
    assert settings.google_sheets_configured is False

    monkeypatch.setenv("ENABLE_GOOGLE_SHEETS_SYNC", "true")
    monkeypatch.setenv("GOOGLE_SERVICE_ACCOUNT_FILE", "/credentials.json")
    monkeypatch.setenv("GOOGLE_SHEETS_SPREADSHEET_ID", "sheet-id")
    settings = Settings.from_env()
    assert settings.google_sheets_enabled is True
    assert settings.google_sheets_configured is True
