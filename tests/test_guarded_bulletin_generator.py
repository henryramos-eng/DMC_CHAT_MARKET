from __future__ import annotations

import unittest

from bulletin.generator import TemplateBulletinGenerator
from bulletin.guarded_generator import GuardedBulletinGenerator
from bulletin.models import BulletinDraft, CommodityInsight, UnifiedBulletinContext
from technical.models import TechnicalRecord


class FakePrimary:
    def generate(self, context: UnifiedBulletinContext) -> BulletinDraft:
        return BulletinDraft(
            title="LLM", classification="Neutral", executive_summary="Resumen",
            highlights=("a", "b", "c"), sections=(), backend="openai:test",
            commodity_insights=tuple(
                CommodityInsight(
                    symbol=symbol,
                    classification="Bajista",
                    analysis_lines=("Contradicción", "factors"),
                    factors=(f"LLM {symbol} 1", f"LLM {symbol} 2", f"LLM {symbol} 3"),
                )
                for symbol in ("QBS", "QSM", "QBO", "CL", "HO")
            ),
        )


def _context() -> UnifiedBulletinContext:
    records = tuple(
        TechnicalRecord(
            date="2026-09-04", symbol=symbol, price_close=100, variation_pct=1,
            ma20=90, ma50=None, rsi=60, macd=1, macd_signal=0.5,
            support=None, resistance=None, trend="Neutral", ema50=80,
        )
        for symbol in ("QBS", "QSM", "QBO", "CL", "HO")
    )
    return UnifiedBulletinContext(
        publication_date="2026-09-04", rag_chunks=(), technical_snapshot=records,
        technical_history={item.symbol: (item,) for item in records},
        technical_source_file="test.xlsx", technical_warnings=(),
    )


class GuardedGeneratorTests(unittest.TestCase):
    def test_technical_analysis_is_deterministic_and_energy_factors_are_guarded(self) -> None:
        context = _context()
        baseline = TemplateBulletinGenerator().generate(context)
        draft = GuardedBulletinGenerator(FakePrimary()).generate(context)
        self.assertEqual(
            draft.insight_for("CL").analysis_lines,
            baseline.insight_for("CL").analysis_lines,
        )
        self.assertEqual(
            draft.insight_for("CL").factors,
            baseline.insight_for("CL").factors,
        )
        self.assertEqual(draft.insight_for("QBS").factors[0], "LLM QBS 1")


if __name__ == "__main__":
    unittest.main()
