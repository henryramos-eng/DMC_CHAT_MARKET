from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from rag.models import Chunk
from rag.vector_index import LocalVectorIndex


class VectorIndexTests(unittest.TestCase):
    def setUp(self) -> None:
        self.chunks = [
            Chunk("a", "China soybean auction", 1, "old.pdf", 0, 3, "2026-09-03"),
            Chunk("b", "French corn conditions", 1, "new.pdf", 4, 7, "2026-09-04"),
            Chunk("c", "Weather outlook", 2, "new.pdf", 8, 11, "2026-09-04"),
        ]
        self.index = LocalVectorIndex(
            np.asarray(
                [[1.0, 0.0], [0.0, 1.0], [0.2, 0.8]],
                dtype=np.float32,
            ),
            self.chunks,
        )

    def test_cosine_search_and_threshold(self) -> None:
        results = self.index.search(np.asarray([0.9, 0.1]), top_k=1, threshold=0.5)
        self.assertEqual(results[0].chunk.chunk_id, "a")

    def test_exact_date_filter_prevents_cross_date_result(self) -> None:
        results = self.index.search(
            np.asarray([0.9, 0.1]),
            top_k=1,
            threshold=-1.0,
            publication_date="2026-09-04",
        )
        self.assertIn(results[0].chunk.chunk_id, {"b", "c"})
        self.assertEqual(self.index.available_dates, ("2026-09-03", "2026-09-04"))

    def test_complete_date_context_is_returned_in_document_order(self) -> None:
        results = self.index.chunks_for_date("2026-09-04")
        self.assertEqual([item.chunk.chunk_id for item in results], ["b", "c"])

    def test_round_trip_preserves_publication_dates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            self.index.save(path)
            loaded = LocalVectorIndex.load(path)
            self.assertEqual([item.chunk_id for item in loaded.chunks], ["a", "b", "c"])
            self.assertEqual(loaded.chunks[0].publication_date, "2026-09-03")


if __name__ == "__main__":
    unittest.main()
