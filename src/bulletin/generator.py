"""Backends de texto intercambiables para el boletín."""

from __future__ import annotations

from collections import Counter
from typing import Protocol

from .models import BulletinDraft, CommodityInsight, UnifiedBulletinContext


class BulletinTextGenerator(Protocol):
    def generate(self, context: UnifiedBulletinContext) -> BulletinDraft: ...


class TemplateBulletinGenerator:
    """Fallback determinista cuando el backend LLM no está disponible."""

    def generate(self, context: UnifiedBulletinContext) -> BulletinDraft:
        counts = Counter(item.trend for item in context.technical_snapshot)
        classification = _aggregate_classification(counts)
        dated = context.publication_date.split("-")
        display_date = f"{dated[2]}/{dated[1]}/{dated[0]}"
        highlights = tuple(_technical_highlight(item) for item in context.technical_snapshot)
        insights = tuple(_template_insight(item) for item in context.technical_snapshot)
        summary = (
            f"Lectura técnica agregada {classification.lower()} para "
            f"{len(context.technical_snapshot)} símbolos. El contexto documental "
            f"incluye {len(context.rag_chunks)} fragmentos del reporte de la misma fecha."
        )
        return BulletinDraft(
            title=f"Boletín técnico de mercados — {display_date}",
            classification=classification,
            executive_summary=summary,
            highlights=highlights,
            sections=(
                {
                    "title": "Fallback determinista",
                    "body": (
                        "Contenido técnico construido con reglas verificables. "
                        "Se utiliza cuando el backend LLM no está disponible."
                    ),
                },
            ),
            backend="template",
            commodity_insights=insights,
        )


def _aggregate_classification(counts: Counter[str]) -> str:
    bullish = counts.get("Alcista", 0)
    bearish = counts.get("Bajista", 0)
    if bullish > bearish and bullish > counts.get("Neutral", 0):
        return "Alcista"
    if bearish > bullish and bearish > counts.get("Neutral", 0):
        return "Bajista"
    return "Neutral"


def _template_insight(item: object) -> CommodityInsight:
    classification = _rule_classification(item)
    if item.price_close is None or item.ma20 is None:
        first = "Historia insuficiente para comparar el precio con MA20."
        relative = "No existe historia suficiente para calcular una referencia MA20."
    elif item.price_close >= item.ma20:
        first = "Precio por encima de MA20; el sesgo técnico permanece positivo."
        relative = "El cierre se mantiene por encima de la media de 20 sesiones."
    else:
        first = "Precio por debajo de MA20; el sesgo técnico es defensivo."
        relative = "El cierre se mantiene por debajo de la media de 20 sesiones."

    if item.rsi is not None and item.rsi >= 70:
        second = f"RSI en {item.rsi:.0f}, dentro de zona de sobrecompra."
    elif item.rsi is not None and item.rsi <= 30:
        second = f"RSI en {item.rsi:.0f}, dentro de zona de sobreventa."
    elif item.macd is not None and item.macd_signal is not None:
        direction = "por encima" if item.macd >= item.macd_signal else "por debajo"
        second = f"RSI en {_number(item.rsi, 0)}; MACD {direction} de su señal."
    else:
        second = "Momentum sin confirmación suficiente."

    factors = (
        relative,
        f"RSI actual de {_number(item.rsi, 1)} y MACD de {_number(item.macd)}.",
        f"Variación diaria de {_percent(item.variation_pct)}; lectura {classification.lower()}.",
    )
    return CommodityInsight(
        symbol=item.symbol,
        classification=classification,
        analysis_lines=(first, second),
        factors=factors,
    )


def _rule_classification(item: object) -> str:
    score = 0
    if item.variation_pct is not None:
        score += 1 if item.variation_pct >= 0 else -1
    if item.price_close is not None and item.ma20 is not None:
        score += 1 if item.price_close >= item.ma20 else -1
    if item.rsi is not None:
        score += 1 if item.rsi >= 50 else -1
    if item.macd is not None and item.macd_signal is not None:
        score += 1 if item.macd >= item.macd_signal else -1
    if score >= 2:
        return "Alcista"
    if score <= -2:
        return "Bajista"
    return "Neutral"


def _technical_highlight(item: object) -> str:
    return (
        f"{item.symbol}: cierre {_number(item.price_close)}, "
        f"variación {_percent(item.variation_pct)}, "
        f"RSI {_number(item.rsi, 1)}, señal {item.trend}."
    )


def _number(value: float | None, decimals: int = 2) -> str:
    return "n.d." if value is None else f"{value:.{decimals}f}"


def _percent(value: float | None) -> str:
    return "n.d." if value is None else f"{value:+.2f}%"
