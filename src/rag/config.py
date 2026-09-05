"""Configuracion independiente de CHAT_MARKET_TEST."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class RAGConfig:
    pdf_dir: Path
    pdf_pattern: str
    index_dir: Path
    openai_api_key: str
    telegram_bot_token: str
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-5-mini"
    chunk_size_tokens: int = 700
    chunk_overlap_tokens: int = 120
    top_k: int = 4
    similarity_threshold: float = 0.22
    allowed_telegram_users: frozenset[int] = frozenset()

    @classmethod
    def from_env(cls) -> "RAGConfig":
        load_dotenv()
        legacy_pdf_path = os.getenv("RAG_PDF_PATH", "").strip()
        default_pdf_dir = (
            str(Path(legacy_pdf_path).parent) if legacy_pdf_path else "data/raw"
        )
        return cls(
            pdf_dir=Path(os.getenv("RAG_PDF_DIR", default_pdf_dir)),
            pdf_pattern=os.getenv(
                "RAG_PDF_PATTERN", "morning_grain_comments_*.pdf"
            ).strip(),
            index_dir=Path(
                os.getenv(
                    "RAG_INDEX_DIR",
                    "data/processed/rag/morning_grain_comments",
                )
            ),
            openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
            telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN", "").strip(),
            embedding_model=os.getenv(
                "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
            ),
            llm_model=os.getenv("OPENAI_LLM_MODEL", "gpt-5-mini"),
            chunk_size_tokens=_env_int("RAG_CHUNK_SIZE_TOKENS", 700),
            chunk_overlap_tokens=_env_int("RAG_CHUNK_OVERLAP_TOKENS", 120),
            top_k=_env_int("RAG_TOP_K", 4),
            similarity_threshold=_env_float("RAG_SIMILARITY_THRESHOLD", 0.22),
            allowed_telegram_users=_parse_user_ids(
                os.getenv("ALLOWED_TELEGRAM_USERS", "")
            ),
        )

    def validate_chunking(self) -> None:
        if self.chunk_size_tokens <= 0:
            raise ValueError("RAG_CHUNK_SIZE_TOKENS debe ser mayor que cero")
        if not 0 <= self.chunk_overlap_tokens < self.chunk_size_tokens:
            raise ValueError(
                "RAG_CHUNK_OVERLAP_TOKENS debe ser no negativo y menor al chunk"
            )
        if not self.pdf_pattern:
            raise ValueError("RAG_PDF_PATTERN no puede estar vacio")


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None else int(value)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value is None else float(value)


def _parse_user_ids(value: str) -> frozenset[int]:
    if not value.strip():
        return frozenset()
    return frozenset(int(item.strip()) for item in value.split(",") if item.strip())
