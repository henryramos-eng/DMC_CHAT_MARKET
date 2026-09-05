"""Entrada del servicio RAG con control tematico y conversacion flexible."""

from __future__ import annotations

from .config import RAGConfig
from .conversation import (
    NOT_FOUND,
    OUT_OF_SCOPE,
    MAX_DATE_OPTIONS,
    PendingQuery,
    RAGReply,
    ResponseMode,
    RAGService as _ConversationRAGService,
    _rank_dates,
)
from .dates import DateQueryError, parse_query_date
from .intents import Intent, classify_intent, normalize
from .openai_client import OpenAIAnswerGenerator
from .retriever import Retriever


NO_RELATED_INFORMATION = (
    "No encontré información relacionada con esa consulta en los reportes cargados. "
    "Puedo ayudarte con granos, mercados, clima agrícola, exportaciones y los países "
    "o acontecimientos mencionados en los boletines."
)

GREETING_SUGGESTIONS = (
    (
        "Dame novedades sobre China",
        "Resumen al día 04092026",
        "¿Qué fechas hay?",
    ),
    (
        "¿Cómo estuvo el clima en Argentina?",
        "Boletín ordenado del 03092026",
        "¿Qué pasó con la soja?",
    ),
    (
        "Tendencia actual del mercado",
        "Novedades sobre exportaciones",
        "Resumen del último reporte",
    ),
    (
        "¿Qué se informó sobre Brasil?",
        "Situación del maíz 01092026",
        "Muéstrame los reportes disponibles",
    ),
)


class RAGService(_ConversationRAGService):
    def __init__(
        self,
        config: RAGConfig,
        *,
        retriever: Retriever | None = None,
        generator: OpenAIAnswerGenerator | None = None,
    ) -> None:
        super().__init__(
            config,
            retriever=retriever,
            generator=generator,
        )
        self._greeting_turn_by_chat: dict[int, int] = {}

    async def handle_reply(self, chat_id: int, text: str) -> RAGReply:
        if classify_intent(text.strip()) is Intent.GREETING:
            self._pending_queries.pop(chat_id, None)
            return self._greeting_reply(chat_id)
        return await super().handle_reply(chat_id, text)

    def _greeting_reply(self, chat_id: int) -> RAGReply:
        turn = self._greeting_turn_by_chat.get(chat_id, 0)
        group_index = (abs(chat_id) + turn) % len(GREETING_SUGGESTIONS)
        self._greeting_turn_by_chat[chat_id] = turn + 1
        suggestions = GREETING_SUGGESTIONS[group_index]
        returning = chat_id in self._greeted_chats
        self._greeted_chats.add(chat_id)

        opening = "Hola nuevamente." if returning else "Hola."
        lines = [
            f"{opening} Puedo consultar los reportes Morning Grain Comments.",
            "Puedes escribir libremente. Algunas ideas:",
        ]
        lines.extend(f"• {suggestion}" for suggestion in suggestions)
        lines.append(
            "Si no indicas una fecha, buscaré el tema y te mostraré únicamente "
            "los reportes con coincidencias relevantes."
        )
        return RAGReply("\n".join(lines))

    async def _handle_pending_text_selection(
        self,
        chat_id: int,
        text: str,
    ) -> RAGReply | None:
        pending = self._pending_queries.get(chat_id)
        if pending is None:
            return None

        if normalize(text) in {"cancelar", "cancelar seleccion", "ninguna"}:
            self._pending_queries.pop(chat_id, None)
            return RAGReply("Selección cancelada. Puedes hacer otra consulta.")

        selected_date: str | None = None
        if text.isdigit() and len(text) <= 2:
            option_number = int(text)
            if 1 <= option_number <= len(pending.date_options):
                selected_date = pending.date_options[option_number - 1]
            else:
                return RAGReply(
                    "Ese número no corresponde a una opción. Elige una fecha de la lista.",
                    pending.date_options,
                )
        else:
            try:
                parsed = parse_query_date(text)
            except DateQueryError:
                parsed = None
            if (
                parsed
                and parsed.publication_date
                and _looks_like_selection_text(parsed.semantic_query)
            ):
                selected_date = parsed.publication_date

        if selected_date is None:
            return None
        return await self.select_date(chat_id, selected_date)

    def _offer_relevant_dates(
        self,
        chat_id: int,
        question: str,
        semantic_query: str,
        mode: ResponseMode,
        *,
        preface: str,
        exclude: tuple[str, ...] = (),
    ) -> RAGReply:
        broad_top_k = max(
            self.retriever.top_k * 4,
            len(self.retriever.available_dates) * 4,
        )
        results = self.retriever.search(
            semantic_query,
            top_k=broad_top_k,
        )
        options = _rank_dates(
            results,
            threshold=self.retriever.threshold,
            exclude=exclude,
        )
        if not options:
            self._pending_queries.pop(chat_id, None)
            return RAGReply(NO_RELATED_INFORMATION)

        return self._offer_known_dates(
            chat_id,
            question,
            semantic_query,
            mode,
            options[:MAX_DATE_OPTIONS],
            preface,
        )


def _looks_like_selection_text(semantic_text: str) -> bool:
    value = normalize(semantic_text).strip(" .,:;-")
    return value in {
        "",
        "elijo",
        "selecciono",
        "quiero",
        "opcion",
        "la opcion",
        "esa",
        "esa fecha",
    }


__all__ = [
    "GREETING_SUGGESTIONS",
    "NOT_FOUND",
    "NO_RELATED_INFORMATION",
    "OUT_OF_SCOPE",
    "PendingQuery",
    "RAGReply",
    "RAGService",
    "ResponseMode",
]
