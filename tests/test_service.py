from __future__ import annotations

import unittest
from pathlib import Path

from rag.config import RAGConfig
from rag.models import Chunk, SearchResult
from rag.service import RAGService, ResponseMode


def result(chunk_id: str, publication_date: str, score: float) -> SearchResult:
    return SearchResult(
        Chunk(
            chunk_id,
            f"contenido {chunk_id}",
            1,
            f"report-{publication_date}.pdf",
            0,
            10,
            publication_date,
        ),
        score,
    )


class FakeRetriever:
    top_k = 4
    threshold = 0.22
    available_dates = ("2026-09-01", "2026-09-03", "2026-09-04")

    def __init__(self) -> None:
        self.search_calls: list[dict[str, object]] = []
        self.cross_date_results = [
            result("china-new", "2026-09-04", 0.91),
            result("china-old", "2026-09-01", 0.85),
            result("weak", "2026-09-03", 0.50),
        ]

    def search(
        self,
        question: str,
        *,
        publication_date: str | None = None,
        top_k: int | None = None,
        threshold: float | None = None,
    ) -> list[SearchResult]:
        self.search_calls.append(
            {
                "question": question,
                "publication_date": publication_date,
                "top_k": top_k,
                "threshold": threshold,
            }
        )
        if publication_date:
            return [result("dated", publication_date, 0.88)]
        return self.cross_date_results

    def chunks_for_date(self, publication_date: str) -> list[SearchResult]:
        return [
            result("full-1", publication_date, 1.0),
            SearchResult(
                Chunk(
                    "full-2",
                    "segunda pagina",
                    2,
                    "report.pdf",
                    10,
                    20,
                    publication_date,
                ),
                1.0,
            ),
        ]


class FakeGenerator:
    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []

    def answer(
        self,
        question: str,
        context: str,
        *,
        response_style: str = "answer",
    ) -> str:
        self.calls.append(
            {
                "question": question,
                "context": context,
                "response_style": response_style,
            }
        )
        return f"respuesta {response_style}"


def config() -> RAGConfig:
    return RAGConfig(
        pdf_dir=Path("data/raw"),
        pdf_pattern="*.pdf",
        index_dir=Path("index"),
        openai_api_key="",
        telegram_bot_token="",
    )


class ServiceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.retriever = FakeRetriever()
        self.generator = FakeGenerator()
        self.service = RAGService(
            config(),
            retriever=self.retriever,
            generator=self.generator,
        )

    async def test_open_question_offers_only_relevant_dates(self) -> None:
        reply = await self.service.handle_reply(1, "Dame novedades sobre China")
        self.assertEqual(reply.date_options, ("2026-09-04", "2026-09-01"))
        self.assertIn("Selecciona un botón", reply.text)
        self.assertEqual(self.generator.calls, [])

    async def test_number_selection_answers_original_question(self) -> None:
        await self.service.handle_reply(2, "Dame novedades sobre China")
        reply = await self.service.handle_reply(2, "2")
        self.assertFalse(reply.date_options)
        self.assertIn("respuesta answer", reply.text)
        self.assertEqual(
            self.retriever.search_calls[-1]["publication_date"],
            "2026-09-01",
        )

    async def test_explicit_summary_uses_complete_report(self) -> None:
        reply = await self.service.handle_reply(3, "Resumen al día 04092026")
        self.assertIn("respuesta summary", reply.text)
        self.assertEqual(
            self.generator.calls[-1]["response_style"],
            ResponseMode.SUMMARY.value,
        )
        self.assertIn("segunda pagina", self.generator.calls[-1]["context"])

    async def test_bulletin_without_date_offers_dates_then_formats(self) -> None:
        offered = await self.service.handle_reply(4, "Quiero un boletín ordenado")
        self.assertTrue(offered.date_options)
        reply = await self.service.select_date(4, offered.date_options[0])
        self.assertIn("respuesta bulletin", reply.text)
        self.assertEqual(
            self.generator.calls[-1]["response_style"],
            ResponseMode.BULLETIN.value,
        )

    async def test_unavailable_date_offers_related_alternatives(self) -> None:
        reply = await self.service.handle_reply(5, "China al 05092026")
        self.assertIn("No tengo un reporte", reply.text)
        self.assertEqual(reply.date_options, ("2026-09-04", "2026-09-01"))


if __name__ == "__main__":
    unittest.main()
