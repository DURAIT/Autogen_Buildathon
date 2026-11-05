from __future__ import annotations

import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Callable, Protocol
from uuid import uuid4

from filelock import FileLock

from .storage import neutralize_spreadsheet_formula


APPROVED_FIELDS = (
    "ticket_id",
    "category",
    "priority",
    "status",
    "confidence",
    "escalation_delivery_state",
    "timestamp_utc",
    "reviewed_at_utc",
    "stage_latencies_ms",
    "total_latency_ms",
)
CATEGORY_ALLOWLIST = {
    "refund", "billing", "account", "login", "subscription", "cancellation",
    "delivery", "technical", "product", "warranty", "general", "other",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _masked_customer_id(value: object) -> str:
    text = str(value or "anonymous")
    if text.lower() == "anonymous":
        return "Anonymous"
    return text[:2] + "***" + text[-2:] if len(text) > 4 else "***"


def _redact_outbound(value: object) -> str:
    text = str(value or "")
    text = re.sub(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", "[EMAIL]", text, flags=re.I)
    text = re.sub(r"(?<!\d)(?:\+?\d[\d ()-]{7,}\d)(?!\d)", "[PHONE]", text)
    text = re.sub(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)", "[PAYMENT_NUMBER]", text)
    text = re.sub(
        r"(?i)\b(?:password|passwd|api[_-]?key|secret|token)\s*[:=]\s*[^\s,;]+",
        "[SECRET]",
        text,
    )
    return neutralize_spreadsheet_formula(text)


def approved_payload(row: dict[str, object]) -> dict[str, str]:
    payload = {field: _redact_outbound(row.get(field, "")) for field in APPROVED_FIELDS}
    category = str(row.get("category", "")).strip().casefold()
    payload["category"] = category if category in CATEGORY_ALLOWLIST else "other"
    payload["masked_customer_id"] = _redact_outbound(
        _masked_customer_id(row.get("user_id", ""))
    )
    return payload


class TicketOutbox:
    """Locked JSONL outbox. Rewrites are atomic; the ticket CSV remains canonical."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._thread_lock = Lock()
        self._file_lock = FileLock(str(self.path) + ".lock")

    def _read_unlocked(self) -> list[dict]:
        if not self.path.exists():
            return []
        records = []
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    records.append(json.loads(line))
        return records

    def _write_unlocked(self, records: list[dict]) -> None:
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(self.path)
        try:
            directory_fd = os.open(self.path.parent, os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except (AttributeError, OSError):
            pass

    def enqueue(self, row: dict[str, object]) -> dict:
        payload = approved_payload(row)
        canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        snapshot_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        ticket_id = str(payload["ticket_id"])
        event_id = hashlib.sha256(f"{ticket_id}:{snapshot_hash}".encode()).hexdigest()
        with self._thread_lock, self._file_lock:
            records = self._read_unlocked()
            for record in records:
                if record["event_id"] == event_id:
                    return record
            now = _utc_now()
            record = {
                "event_id": event_id,
                "ticket_id": ticket_id,
                "snapshot_version": time.time_ns(),
                "snapshot_hash": snapshot_hash,
                "payload": payload,
                "state": "pending",
                "attempts": 0,
                "created_at": now,
                "updated_at": now,
                "sent_at": "",
                "error": "",
                "claim_token": "",
                "lease_expires_at": 0.0,
            }
            records.append(record)
            self._write_unlocked(records)
            return record

    def read_all(self) -> list[dict]:
        with self._thread_lock, self._file_lock:
            return self._read_unlocked()

    def pending(self, limit: int) -> list[dict]:
        return [item for item in self.read_all() if item["state"] == "pending"][:limit]

    def claim_pending(
        self,
        limit: int,
        *,
        lease_seconds: float = 60.0,
        event_ids: set[str] | None = None,
    ) -> list[dict]:
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        now = time.time()
        claimed: list[dict] = []
        with self._thread_lock, self._file_lock:
            records = self._read_unlocked()
            changed = False
            for record in records:
                if record["state"] == "claimed" and float(record.get("lease_expires_at", 0)) <= now:
                    record["state"] = "pending"
                    record["claim_token"] = ""
                    record["lease_expires_at"] = 0.0
                    changed = True
            syncing_ticket_ids = {
                record["ticket_id"] for record in records if record["state"] == "claimed"
            }
            for record in records:
                if (
                    record["state"] != "pending"
                    or (event_ids is not None and record["event_id"] not in event_ids)
                    or record["ticket_id"] in syncing_ticket_ids
                    or len(claimed) >= limit
                ):
                    continue
                record["state"] = "claimed"
                record["claim_token"] = uuid4().hex
                record["lease_expires_at"] = now + lease_seconds
                record["updated_at"] = _utc_now()
                syncing_ticket_ids.add(record["ticket_id"])
                claimed.append(dict(record))
                changed = True
            if changed:
                self._write_unlocked(records)
        return claimed

    def record_attempt(
        self, event_id: str, claim_token: str, *, sent: bool, error: str = ""
    ) -> bool:
        with self._thread_lock, self._file_lock:
            records = self._read_unlocked()
            for record in records:
                if (
                    record["event_id"] != event_id
                    or record.get("claim_token") != claim_token
                    or record["state"] != "claimed"
                    or float(record.get("lease_expires_at", 0)) <= time.time()
                ):
                    continue
                record["attempts"] += 1
                record["updated_at"] = _utc_now()
                record["state"] = "sent" if sent else "claimed"
                record["sent_at"] = _utc_now() if sent else ""
                record["error"] = neutralize_spreadsheet_formula(error)
                if sent:
                    record["claim_token"] = ""
                    record["lease_expires_at"] = 0.0
                self._write_unlocked(records)
                return True
        return False

    def mark_failed(self, event_id: str, claim_token: str, error: str) -> bool:
        with self._thread_lock, self._file_lock:
            records = self._read_unlocked()
            for record in records:
                if (
                    record["event_id"] == event_id
                    and record.get("claim_token") == claim_token
                    and record["state"] == "claimed"
                    and float(record.get("lease_expires_at", 0)) > time.time()
                ):
                    record["state"] = "failed"
                    record["updated_at"] = _utc_now()
                    record["error"] = neutralize_spreadsheet_formula(error)
                    record["claim_token"] = ""
                    record["lease_expires_at"] = 0.0
                    self._write_unlocked(records)
                    return True
        return False

    def retry_failed_event_ids(self, limit: int = 100) -> list[str]:
        reset_event_ids: list[str] = []
        with self._thread_lock, self._file_lock:
            records = self._read_unlocked()
            for record in records:
                if record["state"] != "failed" or len(reset_event_ids) >= limit:
                    continue
                record["state"] = "pending"
                record["updated_at"] = _utc_now()
                record["error"] = ""
                reset_event_ids.append(record["event_id"])
            if reset_event_ids:
                self._write_unlocked(records)
        return reset_event_ids

    def retry_failed(self, limit: int = 100) -> int:
        return len(self.retry_failed_event_ids(limit))


class MirrorTransport(Protocol):
    def mirror(self, event: dict) -> None: ...


class GoogleSheetsMirror:
    def __init__(self, transport: MirrorTransport) -> None:
        self.transport = transport

    def mirror(self, event: dict) -> None:
        self.transport.mirror(event)


class SheetsSynchronizer:
    def __init__(
        self,
        outbox: TicketOutbox,
        mirror: GoogleSheetsMirror,
        *,
        max_attempts: int = 3,
        backoff_seconds: float = 0.25,
        lease_seconds: float = 60.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        self.outbox = outbox
        self.mirror = mirror
        self.max_attempts = max_attempts
        self.backoff_seconds = backoff_seconds
        self.lease_seconds = lease_seconds
        self.sleep = sleep

    def sync_pending(self, limit: int = 100) -> dict[str, int]:
        claimed = self.outbox.claim_pending(limit, lease_seconds=self.lease_seconds)
        return self._sync_claimed(claimed)

    def _sync_claimed(self, claimed: list[dict]) -> dict[str, int]:
        result = {"sent": 0, "failed": 0, "skipped": 0}
        for event in claimed:
            last_error = ""
            for retry in range(self.max_attempts):
                try:
                    self.mirror.mirror(event)
                    finalized = self.outbox.record_attempt(
                        event["event_id"], event["claim_token"], sent=True
                    )
                    result["sent" if finalized else "skipped"] += 1
                    break
                except Exception as exc:
                    last_error = f"{type(exc).__name__}: {exc}"
                    recorded = self.outbox.record_attempt(
                        event["event_id"], event["claim_token"], sent=False, error=last_error
                    )
                    if not recorded:
                        result["skipped"] += 1
                        break
                    if retry + 1 < self.max_attempts:
                        self.sleep(self.backoff_seconds * (2**retry))
            else:
                self.outbox.mark_failed(event["event_id"], event["claim_token"], last_error)
                result["failed"] += 1
        return result

    def retry_failed(self, limit: int = 100) -> dict[str, int]:
        event_ids = set(self.outbox.retry_failed_event_ids(limit))
        claimed = self.outbox.claim_pending(
            limit,
            lease_seconds=self.lease_seconds,
            event_ids=event_ids,
        )
        return self._sync_claimed(claimed)


EVENT_COLUMNS = ("event_id", "ticket_id", "snapshot_version", "snapshot_hash", *APPROVED_FIELDS[1:], "masked_customer_id")
CURRENT_COLUMNS = ("ticket_id", *APPROVED_FIELDS[1:], "masked_customer_id", "snapshot_version", "snapshot_hash")


class GoogleApiSheetsTransport:
    """Google Sheets API transport with event-id and ticket-id upserts using RAW values."""

    def __init__(self, service: object, spreadsheet_id: str) -> None:
        self.service = service
        self.spreadsheet_id = spreadsheet_id

    @classmethod
    def from_service_account(cls, credentials_file: str | Path, spreadsheet_id: str):
        from google.oauth2.service_account import Credentials
        from googleapiclient.discovery import build

        credentials = Credentials.from_service_account_file(
            str(credentials_file), scopes=["https://www.googleapis.com/auth/spreadsheets"]
        )
        return cls(build("sheets", "v4", credentials=credentials, cache_discovery=False), spreadsheet_id)

    def _values(self):
        return self.service.spreadsheets().values()

    def _column(self, tab: str, column: str) -> list[list[str]]:
        response = self._values().get(
            spreadsheetId=self.spreadsheet_id, range=f"'{tab}'!{column}:{column}"
        ).execute()
        return response.get("values", [])

    def _rows(self, tab: str) -> list[list[str]]:
        response = self._values().get(
            spreadsheetId=self.spreadsheet_id, range=f"'{tab}'!A:Z"
        ).execute()
        return response.get("values", [])

    @staticmethod
    def _version_number(value: object) -> int:
        try:
            return int(str(value).split("-", 1)[0])
        except (TypeError, ValueError):
            return -1

    def _append(self, tab: str, values: list[str]) -> None:
        self._values().append(
            spreadsheetId=self.spreadsheet_id,
            range=f"'{tab}'!A:A",
            valueInputOption="RAW",
            insertDataOption="INSERT_ROWS",
            body={"values": [values]},
        ).execute()

    def mirror(self, event: dict) -> None:
        event_ids = self._column("Ticket Events", "A")
        if not event_ids:
            self._append("Ticket Events", list(EVENT_COLUMNS))
        if event["event_id"] not in {str(row[0]) for row in event_ids[1:] if row}:
            values = [event.get(column, event["payload"].get(column, "")) for column in EVENT_COLUMNS]
            self._append("Ticket Events", values)

        current_rows = self._rows("Current Tickets")
        if not current_rows:
            self._append("Current Tickets", list(CURRENT_COLUMNS))
            current_rows = [list(CURRENT_COLUMNS)]
        values = [event.get(column, event["payload"].get(column, "")) for column in CURRENT_COLUMNS]
        existing = next(
            (
                (index, row)
                for index, row in enumerate(current_rows[1:], start=2)
                if row and str(row[0]) == event["ticket_id"]
            ),
            None,
        )
        if existing is None:
            self._append("Current Tickets", values)
        else:
            row_index, existing_row = existing
            version_index = CURRENT_COLUMNS.index("snapshot_version")
            existing_version = (
                existing_row[version_index] if len(existing_row) > version_index else -1
            )
            if self._version_number(event["snapshot_version"]) <= self._version_number(
                existing_version
            ):
                return
            self._values().update(
                spreadsheetId=self.spreadsheet_id,
                range=f"'Current Tickets'!A{row_index}",
                valueInputOption="RAW",
                body={"values": [values]},
            ).execute()
