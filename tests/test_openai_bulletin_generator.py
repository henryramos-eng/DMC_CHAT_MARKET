from __future__ import annotations

import json
import unittest
from pathlib import Path
from types import SimpleNamespace

from bulletin.generator import TemplateBulletinGenerator
from bulletin.models import UnifiedBulletinContext
from bulletin.openai_generator import OpenAIBulletinGenerator
from technical.models import TechnicalRecord


def _record(symbol: str) -> TechnicalRecord:
    return TechnicalRecord(
        date="2026-09-04", symbol=symbol, price_close=100.0,
        variation_pct=1.0, ma20=98.0, ma50=None, rsi=60.0,
        macd=1.0, macd_signal=0.5, support=None, resistance=None,
        trend="Neutral", ema50=95.0,
    )


def _context() -> UnifiedBulletinContext:
    records = tuple(_record(symbol) for symbol in ("QBS", "QSM", "QBO", "CL", "HO"))
    return UnifiedBulletinContext(
        publication_date="2026-09-04",
        rag_chunks=({"text": "China anunció compras de soya.", "page_number": 1},),
        technical_snapshot=records,
        technical_history={item.symbol: (item,) for item in records},
        technical_source_file="test.xlsx",
        technical_warnings=(),
    )


def _payload(symbols: tuple[str, ...]) -> dict[str, object]:
    return {
        "title": "Boletín",
        "classification": "Alcista",
        "executive_summary": "Resumen respaldado por las fuentes.",
        "highlights": ["Uno", "Dos", "Tres"],
        "sections": [{"title": "Mercado", "body": "Contenido"}],
        "commodity_insights": [
            {
                "symbol": symbol,
                "classification": "Alcista",
                "analysis_lines": ["Línea técnica uno.", "Línea técnica dos."],
                "factors": ["Factor uno.", "Factor dos.", "Factor tres."],
            }
            for symbol in symbols
        ],
    }


class FakeResponses:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.kwargs: dict[str, object] = {}

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.kwargs = kwargs
        return SimpleNamespace(
            status="completed",
            output_text=json.dumps(self.payload, ensure_ascii=False),
        )


class OpenAIBulletinGeneratorTests(unittest.TestCase):
    def test_uses_strict_schema_and_builds_five_insights(self) -> None:
        responses = FakeResponses(_payload(("QBS", "QSM", "QBO", "CL", "HO")))
        client = SimpleNamespace(responses=responses)
        generator = OpenAIBulletinGenerator("", "test-model", client=client)
        draft = generator.generate(_context())
        self.assertEqual(draft.backend, "openai:test-model")
        self.assertEqual(len(draft.commodity_insights), 5)
        self.assertFalse(responses.kwargs["store"])
        text = responses.kwargs["text"]
        self.assertTrue(text["format"]["strict"])

    def test_invalid_symbols_use_template_fallback(self) -> None:
        responses = FakeResponses(_payload(("QBS", "QSM", "QBO", "CL", "CL")))
        client = SimpleNamespace(responses=responses)
        generator = OpenAIBulletinGenerator(
            "", "test-model", client=client, fallback=TemplateBulletinGenerator()
        )
        draft = generator.generate(_context())
        self.assertTrue(draft.backend.startswith("template_fallback:"))
        self.assertEqual(len(draft.commodity_insights), 5)


if __name__ == "__main__":
    unittest.main()
