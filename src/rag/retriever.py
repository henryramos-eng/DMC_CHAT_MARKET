"""Recuperacion semantica sobre el indice NumPy."""

from __future__ import annotations

from pathlib import Path

from .models import SearchResult
from .openai_client import OpenAIEmbedder
from .vector_index import LocalVectorIndex


class Retriever:
    def __init__(
        self,
        index_dir: Path,
        embedder: OpenAIEmbedder,
        *,
        top_k: int = 4,
        threshold: float = 0.22,
    ) -> None:
        self.index = LocalVectorIndex.load(index_dir)
        self.embedder = embedder
        self.top_k = top_k
        self.threshold = threshold

    @property
    def available_dates(self) -> tuple[str, ...]:
        return self.index.available_dates

    def chunks_for_date(self, publication_date: str) -> list[SearchResult]:
        return self.index.chunks_for_date(publication_date)

    def search(
        self,
        question: str,
        *,
        publication_date: str | None = None,
        top_k: int | None = None,
        threshold: float | None = None,
    ) -> list[SearchResult]:
        query_vector = self.embedder.embed_query(question)
        return self.index.search(
            query_vector,
            top_k=self.top_k if top_k is None else top_k,
            threshold=self.threshold if threshold is None else threshold,
            publication_date=publication_date,
        )
