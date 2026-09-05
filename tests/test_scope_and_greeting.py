from __future__ import annotations

import unittest

from rag.intents import Intent, classify_intent
from rag.service import NO_RELATED_INFORMATION, RAGService
from test_service import FakeGenerator, FakeRetriever, config


class EmptyRetriever(FakeRetriever):
    def __init__(self) -> None:
        super().__init__()
        self.cross_date_results = []


class ScopeAndGreetingTests(unittest.IsolatedAsyncioTestCase):
    async def test_car_tires_are_rejected_without_dates_or_embeddings(self) -> None:
        retriever = FakeRetriever()
        service = RAGService(
            config(),
            retriever=retriever,
            generator=FakeGenerator(),
        )
        reply = await service.handle_reply(30, "¿Cuántas llantas tiene un carro?")
        self.assertFalse(reply.date_options)
        self.assertIn("únicamente", reply.text)
        self.assertEqual(retriever.search_calls, [])
        self.assertEqual(
            classify_intent("¿Cuántas llantas tiene un carro?"),
            Intent.OUT_OF_SCOPE,
        )

    async def test_no_semantic_evidence_never_falls_back_to_recent_dates(self) -> None:
        service = RAGService(
            config(),
            retriever=EmptyRetriever(),
            generator=FakeGenerator(),
        )
        reply = await service.handle_reply(31, "Háblame del diseño de interiores")
        self.assertEqual(reply.text, NO_RELATED_INFORMATION)
        self.assertFalse(reply.date_options)

    async def test_greeting_rotates_question_suggestions(self) -> None:
        service = RAGService(
            config(),
            retriever=FakeRetriever(),
            generator=FakeGenerator(),
        )
        first = await service.handle_reply(32, "Hola")
        second = await service.handle_reply(32, "Hola")
        self.assertIn("Algunas ideas", first.text)
        self.assertIn("Hola nuevamente", second.text)
        self.assertNotEqual(first.text, second.text)
        self.assertFalse(first.date_options)


if __name__ == "__main__":
    unittest.main()
