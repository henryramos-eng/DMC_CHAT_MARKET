"""Adaptador read-only sobre el indice RAG existente."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rag.vector_index import LocalVectorIndex


class CurrentRAGAdapter:
    def __init__(self, index_dir: Path) -> None:
        self.index = LocalVectorIndex.load(index_dir)

    @property
    def available_dates(self) -> tuple[str, ...]:
        return self.index.available_dates

    def context_for_date(self, publication_date: str) -> tuple[dict[str, Any], ...]:
        results = self.index.chunks_for_date(publication_date)
        if not results:
            raise ValueError(
                f"El RAG actual no contiene el reporte {publication_date}"
            )
        return tuple(
            {
                **result.chunk.to_dict(),
                "score": result.score,
            }
            for result in results
        )
