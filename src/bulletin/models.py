"""Modelos de contexto, contenido y resultado del boletín."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from technical.models import TechnicalRecord


@dataclass(frozen=True)
class UnifiedBulletinContext:
    publication_date: str
    rag_chunks: tuple[dict[str, Any], ...]
    technical_snapshot: tuple[TechnicalRecord, ...]
    technical_history: dict[str, tuple[TechnicalRecord, ...]]
    technical_source_file: str
    technical_warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "publication_date": self.publication_date,
            "rag_context": list(self.rag_chunks),
            "technical_context": {
                "source_file": self.technical_source_file,
                "warnings": list(self.technical_warnings),
                "snapshot": [item.to_dict() for item in self.technical_snapshot],
                "history": {
                    symbol: [item.to_dict() for item in records]
                    for symbol, records in self.technical_history.items()
                },
            },
        }


@dataclass(frozen=True)
class CommodityInsight:
    symbol: str
    classification: str
    analysis_lines: tuple[str, str]
    factors: tuple[str, str, str]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CommodityInsight":
        raw_analysis = tuple(value["analysis_lines"])
        raw_factors = tuple(value["factors"])
        if len(raw_analysis) != 2:
            raise ValueError("Cada commodity requiere exactamente dos líneas de análisis")
        if len(raw_factors) != 3:
            raise ValueError("Cada commodity requiere exactamente tres factores")

        analysis = tuple(
            _normalize_text(
                item,
                180,
                "No existe evidencia suficiente para una conclusión técnica adicional.",
            )
            for item in raw_analysis
        )
        factors = tuple(
            _normalize_text(
                item,
                150,
                "No se identificó un factor adicional verificable en las fuentes.",
            )
            for item in raw_factors
        )
        return cls(
            symbol=str(value["symbol"]).strip().upper(),
            classification=str(value["classification"]).strip(),
            analysis_lines=(analysis[0], analysis[1]),
            factors=(factors[0], factors[1], factors[2]),
        )


@dataclass(frozen=True)
class BulletinDraft:
    title: str
    classification: str
    executive_summary: str
    highlights: tuple[str, ...]
    sections: tuple[dict[str, str], ...]
    backend: str
    commodity_insights: tuple[CommodityInsight, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def insight_for(self, symbol: str) -> CommodityInsight | None:
        normalized = symbol.strip().upper()
        return next(
            (item for item in self.commodity_insights if item.symbol == normalized),
            None,
        )


@dataclass(frozen=True)
class BulletinResult:
    publication_date: str
    images: tuple[Path, ...]
    context_path: Path
    draft_path: Path
    manifest_path: Path
    backend: str


def _normalize_text(value: object, limit: int, fallback: str) -> str:
    text = str(value).strip() or fallback
    if len(text) <= limit:
        return text
    clipped = text[: limit - 1].rsplit(" ", 1)[0].rstrip(" ,;:")
    return clipped + "…"
