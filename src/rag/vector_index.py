"""Indice vectorial local basado en NumPy, sin base de datos externa."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .models import Chunk, SearchResult


class LocalVectorIndex:
    def __init__(self, vectors: np.ndarray, chunks: list[Chunk]) -> None:
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError("La matriz de vectores debe tener dos dimensiones")
        if matrix.shape[0] != len(chunks):
            raise ValueError("La cantidad de vectores no coincide con los chunks")
        if not len(chunks):
            raise ValueError("El indice no puede estar vacio")
        self.vectors = _normalize_rows(matrix)
        self.chunks = chunks

    @property
    def available_dates(self) -> tuple[str, ...]:
        return tuple(
            sorted({chunk.publication_date for chunk in self.chunks if chunk.publication_date})
        )

    def chunks_for_date(self, publication_date: str) -> list[SearchResult]:
        dated_chunks = [
            chunk for chunk in self.chunks if chunk.publication_date == publication_date
        ]
        dated_chunks.sort(
            key=lambda chunk: (
                chunk.source_file.casefold(),
                chunk.page_number,
                chunk.start_token,
            )
        )
        return [SearchResult(chunk=chunk, score=1.0) for chunk in dated_chunks]

    def search(
        self,
        query_vector: np.ndarray,
        *,
        top_k: int = 4,
        threshold: float = 0.22,
        publication_date: str | None = None,
    ) -> list[SearchResult]:
        if top_k <= 0:
            return []
        query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if query.shape[0] != self.vectors.shape[1]:
            raise ValueError("La dimension de la consulta no coincide con el indice")
        norm = float(np.linalg.norm(query))
        if norm == 0:
            return []

        if publication_date:
            candidates = np.asarray(
                [
                    index
                    for index, chunk in enumerate(self.chunks)
                    if chunk.publication_date == publication_date
                ],
                dtype=np.int64,
            )
        else:
            candidates = np.arange(len(self.chunks), dtype=np.int64)
        if not len(candidates):
            return []

        scores = self.vectors @ (query / norm)
        ranked = candidates[np.argsort(scores[candidates])[::-1]][:top_k]
        return [
            SearchResult(chunk=self.chunks[int(index)], score=float(scores[index]))
            for index in ranked
            if float(scores[index]) >= threshold
        ]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(directory / "vectors.npz", vectors=self.vectors)
        _write_json(
            directory / "metadata.json",
            [chunk.to_dict() for chunk in self.chunks],
        )

    @classmethod
    def load(cls, directory: Path) -> "LocalVectorIndex":
        vector_path = directory / "vectors.npz"
        metadata_path = directory / "metadata.json"
        if not vector_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(
                f"Indice incompleto en {directory}. Ejecuta primero rag_index.py"
            )
        with np.load(vector_path, allow_pickle=False) as values:
            vectors = values["vectors"]
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        chunks = [Chunk.from_dict(item) for item in metadata]
        return cls(vectors, chunks)


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("El indice contiene un vector nulo")
    return matrix / norms


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
