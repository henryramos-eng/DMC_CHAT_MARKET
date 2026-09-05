"""Orquestacion conversacional de fechas, recuperacion, LLM y citas."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from enum import Enum

from .config import RAGConfig
from .dates import (
    DateQueryError,
    display_date,
    is_available_dates_question,
    parse_query_date,
    requests_latest_report,
)
from .intents import Intent, classify_intent, normalize
from .models import SearchResult
from .openai_client import OpenAIAnswerGenerator, OpenAIEmbedder
from .retriever import Retriever


NOT_FOUND = "No encontré evidencia suficiente sobre ese tema en el reporte seleccionado."
OUT_OF_SCOPE = "Puedo ayudarte únicamente con consultas relacionadas con los reportes cargados."
MAX_DATE_OPTIONS = 5


class ResponseMode(str, Enum):
    ANSWER = "answer"
    SUMMARY = "summary"
    BULLETIN = "bulletin"


@dataclass(frozen=True)
class RAGReply:
    text: str
    date_options: tuple[str, ...] = ()


@dataclass(frozen=True)
class PendingQuery:
    question: str
    semantic_query: str
    mode: ResponseMode
    date_options: tuple[str, ...]


class RAGService:
    def __init__(
        self,
        config: RAGConfig,
        *,
        retriever: Retriever | None = None,
        generator: OpenAIAnswerGenerator | None = None,
    ) -> None:
        if retriever is None:
            embedder = OpenAIEmbedder(config.openai_api_key, config.embedding_model)
            retriever = Retriever(
                config.index_dir,
                embedder,
                top_k=config.top_k,
                threshold=config.similarity_threshold,
            )
        if generator is None:
            generator = OpenAIAnswerGenerator(
                config.openai_api_key,
                config.llm_model,
            )
        self.retriever = retriever
        self.generator = generator
        self._greeted_chats: set[int] = set()
        self._pending_queries: dict[int, PendingQuery] = {}
        self._last_date_by_chat: dict[int, str] = {}

    async def handle(self, chat_id: int, text: str) -> str:
        """Compatibilidad con FAQ y otros clientes que solo necesitan texto."""
        return (await self.handle_reply(chat_id, text)).text

    async def handle_reply(self, chat_id: int, text: str) -> RAGReply:
        text = text.strip()
        if not text:
            return RAGReply("Escribe una consulta sobre los reportes de granos.")

        pending_reply = await self._handle_pending_text_selection(chat_id, text)
        if pending_reply is not None:
            return pending_reply

        intent = classify_intent(text)
        if intent is Intent.GREETING:
            if chat_id in self._greeted_chats:
                return RAGReply("Hola nuevamente. ¿Qué información deseas consultar?")
            self._greeted_chats.add(chat_id)
            return RAGReply(
                "Hola. Pregunta libremente por un tema, una fecha, un resumen o "
                "un boletín. Si no indicas fecha, te mostraré las mejores opciones."
            )
        if intent is Intent.THANKS:
            return RAGReply("Con gusto.")
        if intent is Intent.GOODBYE:
            return RAGReply("Hasta luego.")
        if intent is Intent.OUT_OF_SCOPE:
            return RAGReply(OUT_OF_SCOPE)

        self._pending_queries.pop(chat_id, None)
        if is_available_dates_question(text):
            return self._offer_known_dates(
                chat_id,
                question="Resumen general del reporte",
                semantic_query="resumen general del reporte",
                mode=ResponseMode.SUMMARY,
                dates=tuple(reversed(self.retriever.available_dates)),
                preface="Estos son los reportes disponibles. Elige uno para ver su resumen:",
                limit=None,
            )
        return await asyncio.to_thread(self._answer_report, chat_id, text)

    async def select_date(self, chat_id: int, publication_date: str) -> RAGReply:
        """Resuelve una seleccion recibida desde un boton de Telegram."""
        return await asyncio.to_thread(
            self._select_date,
            chat_id,
            publication_date,
        )

    def _answer_report(self, chat_id: int, question: str) -> RAGReply:
        available_dates = self.retriever.available_dates
        if not available_dates:
            return RAGReply(
                "El índice no contiene fechas de publicación. Vuelve a generarlo."
            )

        try:
            parsed = parse_query_date(question)
        except DateQueryError as exc:
            return RAGReply(f"{exc} {_available_dates_text(available_dates)}")

        mode = _detect_response_mode(question)
        semantic_query = parsed.semantic_query or "resumen general del reporte"
        selected_date = parsed.publication_date

        if selected_date and selected_date not in available_dates:
            return self._offer_relevant_dates(
                chat_id,
                question,
                semantic_query,
                mode,
                preface=(
                    f"No tengo un reporte publicado el {display_date(selected_date)}. "
                    "Sí encontré estas alternativas relacionadas:"
                ),
            )

        if selected_date:
            return self._answer_with_date(
                chat_id,
                question,
                semantic_query,
                selected_date,
                mode,
            )

        if requests_latest_report(question):
            return self._answer_with_date(
                chat_id,
                question,
                semantic_query,
                available_dates[-1],
                mode,
            )

        if _is_follow_up(question) and chat_id in self._last_date_by_chat:
            return self._answer_with_date(
                chat_id,
                question,
                semantic_query,
                self._last_date_by_chat[chat_id],
                mode,
            )

        if len(available_dates) == 1:
            return self._answer_with_date(
                chat_id,
                question,
                semantic_query,
                available_dates[0],
                mode,
            )

        if mode in (ResponseMode.SUMMARY, ResponseMode.BULLETIN):
            label = "resumen" if mode is ResponseMode.SUMMARY else "boletín"
            return self._offer_known_dates(
                chat_id,
                question,
                semantic_query,
                mode,
                tuple(reversed(available_dates)),
                f"¿De qué fecha deseas el {label}?",
            )

        return self._offer_relevant_dates(
            chat_id,
            question,
            semantic_query,
            mode,
            preface="Encontré información relacionada en estas fechas:",
        )

    def _answer_with_date(
        self,
        chat_id: int,
        question: str,
        semantic_query: str,
        publication_date: str,
        mode: ResponseMode,
    ) -> RAGReply:
        if mode in (ResponseMode.SUMMARY, ResponseMode.BULLETIN):
            results = self.retriever.chunks_for_date(publication_date)
        else:
            results = self.retriever.search(
                semantic_query,
                publication_date=publication_date,
            )

        if not results:
            return self._offer_relevant_dates(
                chat_id,
                question,
                semantic_query,
                mode,
                preface=(
                    f"No encontré evidencia suficiente en el reporte del "
                    f"{display_date(publication_date)}. Puedes revisar estas otras fechas:"
                ),
                exclude=(publication_date,),
            )

        context = _format_context(results)
        dated_question = (
            f"{question}\n"
            f"Fecha del reporte seleccionada: {display_date(publication_date)}"
        )
        answer = self.generator.answer(
            dated_question,
            context,
            response_style=mode.value,
        )
        if not answer or "NO_ENCONTRADO" in answer.upper():
            return RAGReply(NOT_FOUND)

        self._last_date_by_chat[chat_id] = publication_date
        self._pending_queries.pop(chat_id, None)
        return RAGReply(f"{answer}\n\n{_format_sources(results)}")

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
            options = tuple(
                value
                for value in reversed(self.retriever.available_dates)
                if value not in exclude
            )[:MAX_DATE_OPTIONS]
            preface = (
                f"{preface}\nNo hallé una coincidencia temática clara, pero estos "
                "son los reportes más recientes que puedes consultar:"
            )

        return self._offer_known_dates(
            chat_id,
            question,
            semantic_query,
            mode,
            options,
            preface,
        )

    def _offer_known_dates(
        self,
        chat_id: int,
        question: str,
        semantic_query: str,
        mode: ResponseMode,
        dates: tuple[str, ...],
        preface: str,
        *,
        limit: int | None = MAX_DATE_OPTIONS,
    ) -> RAGReply:
        options = dates if limit is None else dates[:limit]
        if not options:
            return RAGReply(NOT_FOUND)
        self._pending_queries[chat_id] = PendingQuery(
            question=question,
            semantic_query=semantic_query,
            mode=mode,
            date_options=options,
        )
        lines = [preface]
        lines.extend(
            f"{index}. {display_date(value)}"
            for index, value in enumerate(options, start=1)
        )
        lines.append("Selecciona un botón o responde con el número o la fecha.")
        return RAGReply("\n".join(lines), options)

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
                and len(text) <= 35
            ):
                selected_date = parsed.publication_date

        if selected_date is None:
            return None
        return await self.select_date(chat_id, selected_date)

    def _select_date(self, chat_id: int, publication_date: str) -> RAGReply:
        pending = self._pending_queries.get(chat_id)
        if pending is None:
            return RAGReply(
                "Esta selección ya no está activa. Haz nuevamente la consulta."
            )
        if publication_date not in pending.date_options:
            return RAGReply(
                "Esa fecha no pertenece a las opciones encontradas.",
                pending.date_options,
            )
        return self._answer_with_date(
            chat_id,
            pending.question,
            pending.semantic_query,
            publication_date,
            pending.mode,
        )


def _rank_dates(
    results: list[SearchResult],
    *,
    threshold: float,
    exclude: tuple[str, ...] = (),
) -> tuple[str, ...]:
    best_by_date: dict[str, float] = {}
    for result in results:
        publication_date = result.chunk.publication_date
        if not publication_date or publication_date in exclude:
            continue
        best_by_date[publication_date] = max(
            result.score,
            best_by_date.get(publication_date, float("-inf")),
        )
    ranked = sorted(
        best_by_date.items(),
        key=lambda item: (item[1], item[0]),
        reverse=True,
    )
    if not ranked:
        return ()

    cutoff = max(threshold, ranked[0][1] - 0.12)
    selected = [value for value, score in ranked if score >= cutoff]
    minimum = min(2, len(ranked))
    if len(selected) < minimum:
        selected = [value for value, _ in ranked[:minimum]]
    return tuple(selected[:MAX_DATE_OPTIONS])


def _detect_response_mode(text: str) -> ResponseMode:
    value = normalize(text)
    if re.search(r"\b(boletin|informe ordenado|reporte ordenado)\b", value):
        return ResponseMode.BULLETIN
    if re.search(r"\b(resumen|resumeme|sintesis)\b", value):
        return ResponseMode.SUMMARY
    return ResponseMode.ANSWER


def _is_follow_up(text: str) -> bool:
    value = normalize(text)
    return bool(
        re.match(
            r"^(y\b|tambien\b|ahora\b|ademas\b|que hay de\b|y que hay de\b)",
            value,
        )
    )


def _available_dates_text(dates: tuple[str, ...]) -> str:
    rendered = ", ".join(display_date(value) for value in dates)
    return f"Fechas disponibles: {rendered}."


def _format_context(results: list[SearchResult]) -> str:
    return "\n\n".join(
        (
            f"[CHUNK {result.chunk.chunk_id} | "
            f"{result.chunk.source_file} | "
            f"fecha {display_date(result.chunk.publication_date)} | "
            f"pagina {result.chunk.page_number}]\n"
            f"{result.chunk.text}"
        )
        for result in results
    )


def _format_sources(results: list[SearchResult]) -> str:
    pages = {
        (
            result.chunk.source_file,
            result.chunk.publication_date,
            result.chunk.page_number,
        )
        for result in results
    }
    lines = [
        f"Fuente: {source}, fecha {display_date(publication_date)}, página {page}."
        for source, publication_date, page in sorted(
            pages,
            key=lambda item: (item[1], item[0].casefold(), item[2]),
        )
    ]
    return "\n".join(lines)
