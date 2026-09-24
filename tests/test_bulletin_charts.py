from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from bulletin.generator import TemplateBulletinGenerator
from bulletin.models import UnifiedBulletinContext
from bulletin.renderer import BulletinRenderer
from technical.models import TechnicalRecord


def record(index: int, symbol: str) -> TechnicalRecord:
    return TechnicalRecord(
        date=f"2026-09-{index + 1:02d}",
        symbol=symbol,
        price_close=100.0 + index,
        variation_pct=1.0,
        ma20=99.0 if index >= 2 else None,
        ma50=None,
        rsi=50.0 + index,
        macd=0.5,
        macd_signal=0.2,
        support=None,
        resistance=None,
        trend="Neutral",
        ema50=98.0,
    )


class BulletinChartsTests(unittest.TestCase):
    def test_renders_exactly_five_commodity_png_pages(self) -> None:
        history = {
            symbol: tuple(record(index, symbol) for index in range(5))
            for symbol in ("QBS", "QSM", "QBO", "CL", "HO")
        }
        context = UnifiedBulletinContext(
            publication_date="2026-09-05",
            rag_chunks=({"text": "contexto"},),
            technical_snapshot=tuple(values[-1] for values in history.values()),
            technical_history=history,
            technical_source_file="test.xlsx",
            technical_warnings=(),
        )
        draft = TemplateBulletinGenerator().generate(context)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "imagen_anterior.png").write_bytes(b"anterior")
            paths = BulletinRenderer().render(context, draft, output)
            self.assertEqual(len(paths), 5)
            self.assertEqual(len(list(output.glob("*.png"))), 5)
            self.assertEqual(
                [path.name for path in paths],
                [
                    "01_soya_qbs.png",
                    "02_harina_soya_qsm.png",
                    "03_aceite_soya_qbo.png",
                    "04_petroleo_wti_cl.png",
                    "05_diesel_ulsd_ho.png",
                ],
            )
            for path in paths:
                self.assertTrue(path.is_file())
                with Image.open(path) as image:
                    self.assertEqual(image.size, (1080, 1920))


if __name__ == "__main__":
    unittest.main()
