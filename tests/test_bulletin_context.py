from __future__ import annotations

import unittest

from bulletin.context_builder import UnifiedContextBuilder
from technical.models import TechnicalDataset, TechnicalRecord


class FakeRAG:
    available_dates = ("2026-09-03", "2026-09-04")

    def context_for_date(self, publication_date: str) -> tuple[dict[str, object], ...]:
        return ({"publication_date": publication_date, "text": "contexto"},)


def record(date: str, symbol: str) -> TechnicalRecord:
    return TechnicalRecord(
        date=date,
        symbol=symbol,
        price_close=100.0,
        variation_pct=1.0,
        ma20=99.0,
        ma50=None,
        rsi=55.0,
        macd=0.5,
        macd_signal=0.2,
        support=None,
        resistance=None,
        trend="Neutral",
    )


class ContextBuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = TechnicalDataset(
            schema_version="1.0",
            source_file="test.xlsx",
            source_sha256="abc",
            generated_at_utc="2026-01-01T00:00:00+00:00",
            records=(
                record("2026-09-02", "AAA"),
                record("2026-09-03", "AAA"),
                record("2026-09-04", "AAA"),
            ),
        )

    def test_uses_latest_common_date_and_preserves_sources(self) -> None:
        context = UnifiedContextBuilder(FakeRAG(), self.dataset).build()
        self.assertEqual(context.publication_date, "2026-09-04")
        self.assertEqual(len(context.rag_chunks), 1)
        self.assertEqual(len(context.technical_history["AAA"]), 3)
        self.assertEqual(context.technical_source_file, "test.xlsx")

    def test_rejects_date_not_shared_by_both_sources(self) -> None:
        with self.assertRaises(ValueError):
            UnifiedContextBuilder(FakeRAG(), self.dataset).build("2026-09-02")


if __name__ == "__main__":
    unittest.main()
