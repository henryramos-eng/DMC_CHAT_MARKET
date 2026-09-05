"""Modelos de datos del pipeline RAG."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ExtractedPage:
    page_number: int
    raw_text: str
    clean_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    page_number: int
    source_file: str
    start_token: int
    end_token: int
    publication_date: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Chunk":
        return cls(**value)


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float
