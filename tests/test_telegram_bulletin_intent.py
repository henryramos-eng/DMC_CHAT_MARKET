from __future__ import annotations

import unittest

from rag.telegram_bot import _select_pending_date, _wants_image_bulletin


class TelegramBulletinIntentTests(unittest.TestCase):
    def test_detects_natural_bulletin_requests(self) -> None:
        self.assertTrue(_wants_image_bulletin("Genera el boletín del 04092026"))
        self.assertTrue(_wants_image_bulletin("Quiero un resumen en imágenes"))
        self.assertFalse(_wants_image_bulletin("Dame novedades sobre China"))

    def test_selects_pending_date_by_number_or_explicit_date(self) -> None:
        dates = ("2026-09-04", "2026-09-01")
        self.assertEqual(_select_pending_date("1", dates), "2026-09-04")
        self.assertEqual(_select_pending_date("01092026", dates), "2026-09-01")
        self.assertIsNone(_select_pending_date("03092026", dates))


if __name__ == "__main__":
    unittest.main()
