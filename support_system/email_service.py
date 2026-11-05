from __future__ import annotations

import smtplib
from email.message import EmailMessage
import re
from typing import Any


class EscalationMailer:
    """SMTP adapter with a safe disabled-by-default gate."""

    def __init__(
        self,
        *,
        enabled: bool,
        host: str = "",
        port: int = 587,
        username: str = "",
        password: str = "",
        sender: str = "",
        recipient: str = "",
        use_tls: bool = True,
    ) -> None:
        self.enabled = enabled
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.sender = sender
        self.recipient = recipient
        self.use_tls = use_tls

    def _configuration_error(self) -> str:
        if not all([self.host, self.sender, self.recipient]):
            return "Email enabled but SMTP configuration is incomplete."
        if any("\r" in value or "\n" in value for value in [self.sender, self.recipient]):
            return "Email configuration contains an invalid sender or recipient address."
        return ""

    @staticmethod
    def _redact_sensitive_text(value: Any) -> str:
        text = str(value or "")
        text = re.sub(
            r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
            "[EMAIL]",
            text,
            flags=re.I,
        )
        text = re.sub(r"(?<!\d)(?:\+?\d[\d ()-]{7,}\d)(?!\d)", "[PHONE]", text)
        text = re.sub(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)", "[PAYMENT_NUMBER]", text)
        return text

    def _deliver(self, message: EmailMessage) -> dict[str, Any]:
        try:
            with smtplib.SMTP(self.host, self.port, timeout=20) as smtp:
                if self.use_tls:
                    smtp.starttls()
                if self.username:
                    smtp.login(self.username, self.password)
                smtp.send_message(message)
            return {"sent": True, "detail": "Email sent."}
        except (OSError, smtplib.SMTPException) as exc:
            return {"sent": False, "detail": f"SMTP error: {type(exc).__name__}"}

    def send(self, **payload: Any) -> dict[str, Any]:
        if not self.enabled:
            return {"sent": False, "detail": "Email disabled; escalation recorded only."}
        configuration_error = self._configuration_error()
        if configuration_error:
            return {"sent": False, "detail": configuration_error}
        message = EmailMessage()
        message["Subject"] = f"Support escalation: {payload.get('ticket_id', 'unknown')}"
        message["From"] = self.sender
        message["To"] = self.recipient
        message.set_content(
            "\n".join(
                [
                    f"Ticket: {payload.get('ticket_id', '')}",
                    f"Category: {self._redact_sensitive_text(payload.get('category', ''))}",
                    f"Priority: {payload.get('priority', '')}",
                    f"Reason: {self._redact_sensitive_text(payload.get('reason', ''))}",
                    "",
                    "Sensitive details remain in the local audit and are not included in this email.",
                ]
            )
        )
        result = self._deliver(message)
        if result["sent"]:
            result["detail"] = "Escalation email sent."
        return result

    def send_review_reply(
        self,
        *,
        ticket_id: str,
        user_id: str,
        original_query: str,
        reply_text: str,
    ) -> dict[str, Any]:
        """Send a human-approved draft to the configured support mailbox."""
        if not self.enabled:
            return {"sent": False, "detail": "Email disabled; review was not sent."}
        configuration_error = self._configuration_error()
        if configuration_error:
            return {"sent": False, "detail": configuration_error}
        clean_reply = reply_text.strip()
        if not clean_reply:
            return {"sent": False, "detail": "Review reply cannot be empty."}
        message = EmailMessage()
        message["Subject"] = f"Human-reviewed support reply: {ticket_id}"
        message["From"] = self.sender
        message["To"] = self.recipient
        message.set_content(
            "\n".join(
                [
                    f"Ticket: {ticket_id}",
                    f"Customer: {self._redact_sensitive_text(user_id)}",
                    "",
                    "Original customer query (common contact/payment patterns redacted):",
                    self._redact_sensitive_text(original_query),
                    "",
                    "Approved reply:",
                    self._redact_sensitive_text(clean_reply),
                ]
            )
        )
        result = self._deliver(message)
        if result["sent"]:
            result["detail"] = f"Human-reviewed reply sent to {self.recipient}."
        return result
