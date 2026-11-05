from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv

from .email_service import EscalationMailer
from .rag import HashingEmbedder, OpenAICompatibleEmbedder


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    model_name: str
    api_key: str
    base_url: str
    confidence_threshold: float
    retrieval_min_score: float
    mcp_server_url: str
    mcp_timeout_seconds: float
    email_enabled: bool
    smtp_host: str
    smtp_port: int
    smtp_username: str
    smtp_password: str
    smtp_sender: str
    support_email: str
    google_sheets_enabled: bool
    google_service_account_file: str
    google_sheets_spreadsheet_id: str
    embedding_backend: str
    embedding_base_url: str
    embedding_model: str
    embedding_dimension: int
    embedding_api_key: str
    embedding_timeout_seconds: float
    llm_rerank_enabled: bool

    @property
    def google_sheets_configured(self) -> bool:
        return bool(
            self.google_sheets_enabled
            and self.google_service_account_file
            and self.google_sheets_spreadsheet_id
        )

    def __post_init__(self) -> None:
        if not math.isfinite(self.confidence_threshold) or not 0 <= self.confidence_threshold <= 1:
            raise ValueError("CONFIDENCE_THRESHOLD must be a finite value between 0 and 1")
        if not math.isfinite(self.retrieval_min_score) or not 0 <= self.retrieval_min_score <= 1:
            raise ValueError("RETRIEVAL_MIN_SCORE must be a finite value between 0 and 1")
        if not math.isfinite(self.mcp_timeout_seconds) or self.mcp_timeout_seconds <= 0:
            raise ValueError("MCP_TIMEOUT_SECONDS must be a positive finite value")
        if not 1 <= self.smtp_port <= 65535:
            raise ValueError("SMTP_PORT must be between 1 and 65535")
        if self.embedding_backend not in {"hashing", "openai-compatible"}:
            raise ValueError("EMBEDDING_BACKEND must be hashing or openai-compatible")
        if self.embedding_dimension <= 0:
            raise ValueError("EMBEDDING_DIMENSION must be positive")
        if not math.isfinite(self.embedding_timeout_seconds) or self.embedding_timeout_seconds <= 0:
            raise ValueError("EMBEDDING_TIMEOUT_SECONDS must be a positive finite value")
        if self.embedding_backend == "openai-compatible" and not (
            self.embedding_base_url and self.embedding_model
        ):
            raise ValueError(
                "OpenAI-compatible embeddings require EMBEDDING_BASE_URL and EMBEDDING_MODEL"
            )

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        return cls(
            data_dir=Path(os.getenv("DATA_DIR", "data")),
            model_name=os.getenv("MODEL_NAME", "gpt-4o-mini"),
            api_key=os.getenv("OPENAI_API_KEY", ""),
            base_url=os.getenv("OPENAI_BASE_URL", ""),
            confidence_threshold=float(os.getenv("CONFIDENCE_THRESHOLD", "0.70")),
            retrieval_min_score=float(os.getenv("RETRIEVAL_MIN_SCORE", "0.20")),
            mcp_server_url=os.getenv("MCP_SERVER_URL", "http://127.0.0.1:8765/mcp"),
            mcp_timeout_seconds=float(os.getenv("MCP_TIMEOUT_SECONDS", "20")),
            email_enabled=os.getenv("ENABLE_ESCALATION_EMAIL", "false").lower() == "true",
            smtp_host=os.getenv("SMTP_HOST", ""),
            smtp_port=int(os.getenv("SMTP_PORT", "587")),
            smtp_username=os.getenv("SMTP_USERNAME", ""),
            smtp_password=os.getenv("SMTP_PASSWORD", ""),
            smtp_sender=os.getenv("SMTP_SENDER", ""),
            support_email=os.getenv("SUPPORT_EMAIL", "titu.durais@gmail.com"),
            google_sheets_enabled=os.getenv("ENABLE_GOOGLE_SHEETS_SYNC", "false").lower()
            == "true",
            google_service_account_file=os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE", ""),
            google_sheets_spreadsheet_id=os.getenv("GOOGLE_SHEETS_SPREADSHEET_ID", ""),
            embedding_backend=os.getenv("EMBEDDING_BACKEND", "hashing").strip().lower(),
            embedding_base_url=os.getenv(
                "EMBEDDING_BASE_URL", os.getenv("OPENAI_BASE_URL", "")
            ),
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            embedding_dimension=int(os.getenv("EMBEDDING_DIMENSION", "1536")),
            embedding_api_key=os.getenv(
                "EMBEDDING_API_KEY", os.getenv("OPENAI_API_KEY", "")
            ),
            embedding_timeout_seconds=float(os.getenv("EMBEDDING_TIMEOUT_SECONDS", "30")),
            llm_rerank_enabled=os.getenv("ENABLE_LLM_RERANK", "true").lower() == "true",
        )

    def build_embedder(self):
        if self.embedding_backend == "hashing":
            return HashingEmbedder()
        return OpenAICompatibleEmbedder(
            base_url=self.embedding_base_url,
            model=self.embedding_model,
            dimension=self.embedding_dimension,
            api_key=self.embedding_api_key,
            timeout=self.embedding_timeout_seconds,
        )

    def build_model_client(self) -> OpenAIChatCompletionClient:
        kwargs = {"model": self.model_name, "temperature": 0.1}
        if self.base_url:
            kwargs.update(
                {
                    "base_url": self.base_url,
                    "api_key": self.api_key or "local-model",
                    "model_info": {
                        "vision": False,
                        "function_calling": True,
                        "json_output": True,
                        "family": "unknown",
                        "structured_output": True,
                    },
                }
            )
        elif self.api_key:
            kwargs["api_key"] = self.api_key
        else:
            raise ValueError(
                "Set OPENAI_API_KEY, or configure OPENAI_BASE_URL for an OpenAI-compatible local server."
            )
        return OpenAIChatCompletionClient(**kwargs)

    def build_mailer(self) -> EscalationMailer:
        return EscalationMailer(
            enabled=self.email_enabled,
            host=self.smtp_host,
            port=self.smtp_port,
            username=self.smtp_username,
            password=self.smtp_password,
            sender=self.smtp_sender,
            recipient=self.support_email,
        )
