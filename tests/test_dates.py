from __future__ import annotations

import unittest

from rag.dates import (
    DateQueryError,
    display_date,
    is_available_dates_question,
    parse_query_date,
    publication_date_from_filename,
    requests_latest_report,
)


class DateTests(unittest.TestCase):
    def test_date_is_read_from_report_filename(self) -> None:
        value = publication_date_from_filename(
            "morning_grain_comments_03092026.pdf"
        )
        self.assertEqual(value, "2026-09-03")

    def test_short_query_accepts_common_date_formats(self) -> None:
        questions = (
            "clima argentina 03092026",
            "clima argentina 03/09/2026",
            "clima argentina 03-09-2026",
            "clima argentina 2026-09-03",
            "clima argentina 3 de septiembre de 2026",
        )
        for question in questions:
            with self.subTest(question=question):
                parsed = parse_query_date(question)
                self.assertEqual(parsed.publication_date, "2026-09-03")
                self.assertEqual(parsed.semantic_query, "clima argentina")

    def test_invalid_and_multiple_dates_are_rejected(self) -> None:
        with self.assertRaises(DateQueryError):
            parse_query_date("clima 31022026")
        with self.assertRaises(DateQueryError):
            parse_query_date("compara 03092026 con 04092026")

    def test_date_listing_and_latest_intents(self) -> None:
        self.assertTrue(is_available_dates_question("¿Qué fechas hay?"))
        self.assertTrue(requests_latest_report("tendencia actual del mercado"))
        self.assertEqual(display_date("2026-09-03"), "03/09/2026")


if __name__ == "__main__":
    unittest.main()
