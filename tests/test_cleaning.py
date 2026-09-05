from __future__ import annotations

import unittest

from rag.cleaning import clean_pages, clean_text
from rag.models import ExtractedPage


class CleaningTests(unittest.TestCase):
    def test_repairs_common_pdf_artifacts(self) -> None:
        raw = "� China�s sales were contin-\nually strong.\nanalyst@example.com"
        cleaned = clean_text(raw)
        self.assertIn("- China's sales were continually strong.", cleaned)
        self.assertNotIn("@", cleaned)
        self.assertNotIn("�", cleaned)

    def test_report_preserves_counts_and_empty_pages(self) -> None:
        pages = [
            ExtractedPage(1, "Market text"),
            ExtractedPage(2, "   "),
        ]
        cleaned, report = clean_pages(pages)
        self.assertEqual(cleaned[0].clean_text, "Market text")
        self.assertEqual(report.empty_pages, (2,))
        self.assertEqual(report.pages_with_raw_text, 1)


if __name__ == "__main__":
    unittest.main()
