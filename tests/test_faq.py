from __future__ import annotations

import json
import unittest
from pathlib import Path


class FAQTests(unittest.TestCase):
    def test_faq_is_a_dated_twelve_question_evaluation_set(self) -> None:
        root = Path(__file__).resolve().parents[1]
        path = root / "data" / "faq" / "morning_grain_comments_04092026.json"
        faq = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(len(faq), 12)
        self.assertEqual(len({item["id"] for item in faq}), 12)
        self.assertTrue(all(item["expected_page"] == 1 for item in faq))
        self.assertTrue(
            all(item["publication_date"] == "2026-09-04" for item in faq)
        )


if __name__ == "__main__":
    unittest.main()
