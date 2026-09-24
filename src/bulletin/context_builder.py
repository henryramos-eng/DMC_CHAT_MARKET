"""Union determinista entre el RAG existente y el JSON tecnico."""

from __future__ import annotations

from collections.abc import Iterable

from technical.models import TechnicalDataset

from .models import UnifiedBulletinContext
from .rag_adapter import CurrentRAGAdapter


class UnifiedContextBuilder:
    def __init__(
        self,
        rag: CurrentRAGAdapter,
        technical: TechnicalDataset,
        *,
        history_days: int = 60,
    ) -> None:
        self.rag = rag
        self.technical = technical
        self.history_days = history_days

    @property
    def available_dates(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.rag.available_dates).intersection(self.technical.dates)))

    def build(
        self,
        publication_date: str | None = None,
        *,
        symbols: Iterable[str] | None = None,
    ) -> UnifiedBulletinContext:
        common_dates = self.available_dates
        if not common_dates:
            raise ValueError("No hay fechas comunes entre el RAG y el Excel tecnico")
        selected_date = publication_date or common_dates[-1]
        if selected_date not in common_dates:
            raise ValueError(
                f"La fecha {selected_date} no existe simultaneamente en RAG y Excel. "
                f"Fechas comunes: {', '.join(common_dates)}"
            )

        requested = (
            {value.strip().upper() for value in symbols}
            if symbols is not None
            else set(self.technical.symbols)
        )
        snapshot = tuple(
            item
            for item in self.technical.records_for_date(selected_date)
            if item.symbol in requested
        )
        if not snapshot:
            raise ValueError("No hay simbolos tecnicos para la seleccion solicitada")
        missing = requested.difference(item.symbol for item in snapshot)
        if missing:
            raise ValueError(
                "Simbolos sin datos para la fecha: " + ", ".join(sorted(missing))
            )

        history = {
            item.symbol: self.technical.history_for_symbol(
                item.symbol,
                through_date=selected_date,
                limit=self.history_days,
            )
            for item in snapshot
        }
        return UnifiedBulletinContext(
            publication_date=selected_date,
            rag_chunks=self.rag.context_for_date(selected_date),
            technical_snapshot=snapshot,
            technical_history=history,
            technical_source_file=self.technical.source_file,
            technical_warnings=self.technical.warnings,
        )
