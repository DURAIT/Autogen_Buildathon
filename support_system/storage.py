from __future__ import annotations

import csv
import json
import logging
import math
import os
from functools import wraps
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

from filelock import FileLock

from .models import Classification, Resolution

logger = logging.getLogger(__name__)


def emit_snapshot_after_success(method):
    @wraps(method)
    def wrapped(self, ticket_id, *args, **kwargs):
        changed = method(self, ticket_id, *args, **kwargs)
        if changed:
            snapshot = next(
                (row for row in self.read_all() if row.get("ticket_id") == ticket_id), None
            )
            self._emit(snapshot)
        return changed

    return wrapped


def neutralize_spreadsheet_formula(value: object) -> str:
    """Prevent CSV cells from becoming formulas in spreadsheet applications."""
    text = str(value)
    if text.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + text
    return text


class TicketLogger:
    FIELDNAMES = [
        "ticket_id",
        "timestamp_utc",
        "user_id",
        "query",
        "category",
        "urgency",
        "sentiment",
        "priority",
        "summary",
        "is_product_related",
        "answer",
        "confidence",
        "sources",
        "status",
        "escalation_reason",
        "escalation_delivery_id",
        "escalation_delivery_state",
        "escalation_email_status",
        "review_reply",
        "review_delivery_id",
        "review_target_status",
        "review_email_status",
        "reviewed_at_utc",
        "stage_latencies_ms",
        "total_latency_ms",
    ]

    def __init__(self, path: str | Path, event_sink=None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._file_lock = FileLock(str(self.path.with_suffix(self.path.suffix + ".lock")))
        self._event_sink = event_sink

    def _emit(self, row: dict[str, str] | None) -> None:
        if row is None or self._event_sink is None:
            return
        try:
            self._event_sink(dict(row))
        except Exception:
            logger.exception("Ticket mirror enqueue failed; local CSV remains authoritative")

    def append(
        self,
        *,
        ticket_id: str,
        user_id: str,
        query: str,
        classification: Classification,
        resolution: Resolution,
        status: str,
        escalation_reason: str,
        escalation_delivery_id: str = "",
        escalation_delivery_state: str = "",
        escalation_email_status: str = "",
    ) -> None:
        raw_row = {
            "ticket_id": ticket_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "user_id": user_id,
            "query": query,
            "category": classification.category,
            "urgency": classification.urgency,
            "sentiment": classification.sentiment,
            "priority": classification.priority,
            "summary": classification.summary,
            "is_product_related": str(classification.is_product_related).lower(),
            "answer": resolution.answer,
            "confidence": f"{resolution.confidence:.4f}",
            "sources": json.dumps(resolution.source_filenames, ensure_ascii=False),
            "status": status,
            "escalation_reason": escalation_reason,
            "escalation_delivery_id": escalation_delivery_id,
            "escalation_delivery_state": escalation_delivery_state,
            "escalation_email_status": escalation_email_status,
            "review_reply": "",
            "review_delivery_id": "",
            "review_target_status": "",
            "review_email_status": "",
            "reviewed_at_utc": "",
            "stage_latencies_ms": "",
            "total_latency_ms": "",
        }
        row = {key: neutralize_spreadsheet_formula(value) for key, value in raw_row.items()}
        with self._lock, self._file_lock:
            existing_rows, existing_fields = self._read_unlocked()
            if existing_fields and existing_fields != self.FIELDNAMES:
                self._write_all_unlocked(existing_rows)
            exists = self.path.exists() and self.path.stat().st_size > 0
            with self.path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.FIELDNAMES)
                if not exists:
                    writer.writeheader()
                writer.writerow(row)
                handle.flush()
                os.fsync(handle.fileno())
        self._emit(row)

    def _read_unlocked(self) -> tuple[list[dict[str, str]], list[str]]:
        if not self.path.exists() or self.path.stat().st_size == 0:
            return [], []
        with self.path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            return list(reader), list(reader.fieldnames or [])

    def _write_all_unlocked(self, rows: list[dict[str, str]]) -> None:
        temporary_path = self.path.with_suffix(self.path.suffix + ".tmp")
        with temporary_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=self.FIELDNAMES,
                extrasaction="ignore",
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        field: neutralize_spreadsheet_formula(row.get(field, ""))
                        for field in self.FIELDNAMES
                    }
                )
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(self.path)

    @staticmethod
    def _validate_review(review_reply: str, final_status: str) -> str:
        if final_status not in {"open", "resolved"}:
            raise ValueError("final_status must be open or resolved")
        clean_reply = " ".join(review_reply.split())
        if not clean_reply:
            raise ValueError("review_reply cannot be empty")
        if len(clean_reply) > 8_000:
            raise ValueError("review_reply exceeds 8,000 characters")
        return clean_reply

    def claim_review_delivery(
        self,
        ticket_id: str,
        *,
        delivery_id: str,
        review_reply: str,
        final_status: str,
    ) -> bool:
        clean_reply = self._validate_review(review_reply, final_status)
        snapshot = None
        with self._lock, self._file_lock:
            rows, _ = self._read_unlocked()
            for row in rows:
                if row.get("ticket_id") != ticket_id:
                    continue
                current = row.get("review_email_status", "")
                if current.startswith("sending:") or current.startswith("sent:"):
                    break
                row["review_reply"] = neutralize_spreadsheet_formula(clean_reply)
                row["review_delivery_id"] = delivery_id
                row["review_target_status"] = final_status
                row["review_email_status"] = f"sending:{delivery_id}"
                self._write_all_unlocked(rows)
                snapshot = dict(row)
                break
        self._emit(snapshot)
        return snapshot is not None

    @emit_snapshot_after_success
    def finalize_review_delivery(
        self,
        ticket_id: str,
        *,
        delivery_id: str,
        sent: bool,
        detail: str,
    ) -> bool:
        with self._lock, self._file_lock:
            rows, _ = self._read_unlocked()
            for row in rows:
                if row.get("ticket_id") != ticket_id:
                    continue
                if row.get("review_delivery_id") != delivery_id:
                    return False
                if row.get("review_email_status") != f"sending:{delivery_id}":
                    return False
                row["review_email_status"] = neutralize_spreadsheet_formula(
                    ("sent:" if sent else "failed:") + detail
                )
                if sent:
                    row["status"] = row.get("review_target_status", "resolved")
                    row["reviewed_at_utc"] = datetime.now(timezone.utc).isoformat().replace(
                        "+00:00", "Z"
                    )
                self._write_all_unlocked(rows)
                return True
            return False

    @emit_snapshot_after_success
    def claim_escalation_delivery(self, ticket_id: str, delivery_id: str) -> bool:
        with self._lock, self._file_lock:
            rows, _ = self._read_unlocked()
            if any(
                row.get("escalation_delivery_id") == delivery_id
                and row.get("escalation_delivery_state") in {"sending", "sent"}
                for row in rows
            ):
                return False
            for row in rows:
                if row.get("ticket_id") == ticket_id:
                    row["escalation_delivery_state"] = "sending"
                    self._write_all_unlocked(rows)
                    return True
            return False

    @emit_snapshot_after_success
    def finalize_escalation_delivery(
        self,
        ticket_id: str,
        *,
        sent: bool,
        detail: str,
    ) -> bool:
        with self._lock, self._file_lock:
            rows, _ = self._read_unlocked()
            for row in rows:
                if row.get("ticket_id") != ticket_id:
                    continue
                if row.get("escalation_delivery_state") != "sending":
                    return False
                row["escalation_delivery_state"] = "sent" if sent else "failed"
                row["escalation_email_status"] = neutralize_spreadsheet_formula(detail)
                self._write_all_unlocked(rows)
                return True
            return False

    @emit_snapshot_after_success
    def mark_escalation_duplicate(self, ticket_id: str) -> bool:
        with self._lock, self._file_lock:
            rows, _ = self._read_unlocked()
            for row in rows:
                if row.get("ticket_id") == ticket_id:
                    row["escalation_delivery_state"] = "duplicate_suppressed"
                    row["escalation_email_status"] = "duplicate_suppressed"
                    self._write_all_unlocked(rows)
                    return True
            return False

    @emit_snapshot_after_success
    def update_ticket(
        self,
        ticket_id: str,
        *,
        status: str,
        review_reply: str,
        review_email_status: str,
    ) -> bool:
        if status not in {"open", "resolved", "escalated"}:
            raise ValueError("status must be open, resolved, or escalated")
        clean_reply = " ".join(review_reply.split())
        if not clean_reply:
            raise ValueError("review_reply cannot be empty")
        if len(clean_reply) > 8_000:
            raise ValueError("review_reply exceeds 8,000 characters")
        with self._lock, self._file_lock:
            rows, _ = self._read_unlocked()
            updated = False
            for row in rows:
                if row.get("ticket_id") != ticket_id:
                    continue
                row["status"] = status
                row["review_reply"] = neutralize_spreadsheet_formula(clean_reply)
                row["review_email_status"] = neutralize_spreadsheet_formula(review_email_status)
                row["reviewed_at_utc"] = datetime.now(timezone.utc).isoformat().replace(
                    "+00:00", "Z"
                )
                updated = True
                break
            if updated:
                self._write_all_unlocked(rows)
            return updated

    @emit_snapshot_after_success
    def update_observability(
        self,
        ticket_id: str,
        stage_latencies_ms: dict[str, float],
        total_latency_ms: float,
    ) -> bool:
        normalized = {name: float(value) for name, value in stage_latencies_ms.items()}
        total = float(total_latency_ms)
        if any(not math.isfinite(value) or value < 0 for value in normalized.values()) or (
            not math.isfinite(total) or total < 0
        ):
            raise ValueError("latencies must be finite and non-negative")
        serialized = neutralize_spreadsheet_formula(
            json.dumps(normalized, ensure_ascii=False, sort_keys=True, allow_nan=False)
        )
        with self._lock, self._file_lock:
            rows, _ = self._read_unlocked()
            for row in rows:
                if row.get("ticket_id") != ticket_id:
                    continue
                row["stage_latencies_ms"] = serialized
                row["total_latency_ms"] = neutralize_spreadsheet_formula(total)
                self._write_all_unlocked(rows)
                return True
            return False

    def read_all(self) -> list[dict[str, str]]:
        if not self.path.exists():
            return []
        with self._lock, self._file_lock:
            rows, fields = self._read_unlocked()
            if fields and fields != self.FIELDNAMES:
                self._write_all_unlocked(rows)
                rows, _ = self._read_unlocked()
            return rows
