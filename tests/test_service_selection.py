from __future__ import annotations

import unittest

from rag.service import RAGService
from test_service import FakeGenerator, FakeRetriever, config


class SelectionConversationTests(unittest.IsolatedAsyncioTestCase):
    async def test_new_dated_question_replaces_pending_selection(self) -> None:
        generator = FakeGenerator()
        service = RAGService(
            config(),
            retriever=FakeRetriever(),
            generator=generator,
        )
        await service.handle_reply(20, "Dame novedades sobre China")
        reply = await service.handle_reply(20, "Resumen al día 04092026")
        self.assertIn("respuesta summary", reply.text)
        self.assertEqual(generator.calls[-1]["response_style"], "summary")


if __name__ == "__main__":
    unittest.main()
