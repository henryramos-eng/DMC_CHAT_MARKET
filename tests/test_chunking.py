from __future__ import annotations

import unittest

from rag.chunking import chunk_pages
from rag.models import ExtractedPage


class ChunkingTests(unittest.TestCase):
    def test_chunks_never_cross_page_boundaries_and_keep_date(self) -> None:
        pages = [
            ExtractedPage(1, "", "soybean " * 80),
            ExtractedPage(2, "", "disclaimer " * 80),
        ]
        chunks = chunk_pages(
            pages,
            "morning_grain_comments_03092026.pdf",
            publication_date="2026-09-03",
            chunk_size_tokens=30,
            overlap_tokens=5,
        )
        self.assertGreater(len(chunks), 2)
        self.assertTrue(all(chunk.page_number in (1, 2) for chunk in chunks))
        self.assertTrue(all(chunk.publication_date == "2026-09-03" for chunk in chunks))
        self.assertFalse(
            any("soybean" in chunk.text and "disclaimer" in chunk.text for chunk in chunks)
        )


if __name__ == "__main__":
    unittest.main()
