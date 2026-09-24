"""Controles de evidencia sobre el borrador editorial generado por LLM."""

from __future__ import annotations

from dataclasses import replace

from .generator import BulletinTextGenerator, TemplateBulletinGenerator
from .models import BulletinDraft, CommodityInsight, UnifiedBulletinContext


SOY_SYMBOLS = {"QBS", "QSM", "QBO"}


class GuardedBulletinGenerator:
    """Combina redacción LLM con conclusiones técnicas deterministas."""

    def __init__(
        self,
        primary: BulletinTextGenerator,
        deterministic: TemplateBulletinGenerator | None = None,
    ) -> None:
        self.primary = primary
        self.deterministic = deterministic or TemplateBulletinGenerator()

    def generate(self, context: UnifiedBulletinContext) -> BulletinDraft:
        draft = self.primary.generate(context)
        if not draft.backend.startswith("openai:"):
            return draft

        baseline = self.deterministic.generate(context)
        baseline_by_symbol = {
            item.symbol: item for item in baseline.commodity_insights
        }
        llm_by_symbol = {item.symbol: item for item in draft.commodity_insights}
        guarded: list[CommodityInsight] = []
        for symbol in ("QBS", "QSM", "QBO", "CL", "HO"):
            technical = baseline_by_symbol[symbol]
            editorial = llm_by_symbol.get(symbol)
            factors = (
                editorial.factors
                if editorial is not None and symbol in SOY_SYMBOLS
                else technical.factors
            )
            guarded.append(
                CommodityInsight(
                    symbol=symbol,
                    classification=technical.classification,
                    analysis_lines=technical.analysis_lines,
                    factors=factors,
                )
            )
        return replace(
            draft,
            commodity_insights=tuple(guarded),
            sections=draft.sections
            + (
                {
                    "title": "Control de evidencia",
                    "body": (
                        "Clasificación y análisis calculados desde el Excel. "
                        "Factores energéticos limitados a evidencia técnica porque "
                        "el reporte documental es de granos."
                    ),
                },
            ),
        )
