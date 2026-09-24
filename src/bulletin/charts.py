"""Cinco boletines verticales, uno por commodity, generados con Python."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

from .models import BulletinDraft, CommodityInsight, UnifiedBulletinContext


INK = "#06264C"
TEAL = "#004D64"
TEAL_DARK = "#00394F"
PRICE = "#087A61"
BLUE = "#1468D4"
ORANGE = "#F28B00"
GREEN = "#079447"
RED = "#E51D2A"
AMBER = "#D79B22"
LIGHT = "#F7FAFA"
WHITE = "#FFFFFF"
MUTED = "#607783"
GRID = "#D7E3E8"

ORDER = ("QBS", "QSM", "QBO", "CL", "HO")
META = {
    "QBS": ("Complejo Soya", "Soya", "CBOT · CENTAVOS POR BUSHEL", "rsi", "01_soya_qbs.png"),
    "QSM": ("Harina de Soya", "Harina de Soya", "CBOT · USD POR TONELADA CORTA", "rsi", "02_harina_soya_qsm.png"),
    "QBO": ("Aceite de Soya", "Aceite de Soya", "CBOT · CENTAVOS POR LIBRA", "rsi", "03_aceite_soya_qbo.png"),
    "CL": ("Petróleo WTI", "Petróleo WTI", "NYMEX · USD POR BARRIL", "macd", "04_petroleo_wti_cl.png"),
    "HO": ("Diésel / ULSD", "Diésel / ULSD", "NY HARBOR · USD POR GALÓN", "macd", "05_diesel_ulsd_ho.png"),
}


def render_pages(context: UnifiedBulletinContext, draft: BulletinDraft, output_dir: Path) -> tuple[Path, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_png in output_dir.glob("*.png"):
        old_png.unlink()
    snapshot = {item.symbol: item for item in context.technical_snapshot}
    missing = [symbol for symbol in ORDER if symbol not in snapshot]
    if missing:
        raise ValueError("Faltan commodities requeridos: " + ", ".join(missing))
    paths: list[Path] = []
    for page_number, symbol in enumerate(ORDER, start=1):
        title, name, market, indicator, filename = META[symbol]
        path = output_dir / filename
        _render_commodity(context, draft, snapshot[symbol], page_number, title, name, market, indicator, path)
        paths.append(path)
    return tuple(paths)


def _new_canvas() -> tuple[plt.Figure, plt.Axes]:
    figure = plt.figure(figsize=(10.8, 19.2), dpi=100, facecolor=LIGHT)
    canvas = figure.add_axes([0, 0, 1, 1])
    canvas.set_axis_off()
    canvas.set_xlim(0, 1)
    canvas.set_ylim(0, 1)
    canvas.set_autoscale_on(False)
    return figure, canvas


def _txt(canvas: plt.Axes, x: float, y: float, value: object, **kwargs: object) -> object:
    """Todo texto del lienzo usa coordenadas normalizadas, igual que las cajas."""
    return canvas.text(x, y, str(value), transform=canvas.transAxes, **kwargs)


def _render_commodity(context, draft, item, page_number, title, name, market, indicator, path) -> None:
    insight = draft.insight_for(item.symbol)
    sentiment = insight.classification if insight else _rule_sentiment(item)
    figure, canvas = _new_canvas()
    _hero(canvas, title, context.publication_date, page_number)
    _instrument_header(canvas, item.symbol, name, market, sentiment)
    _price_summary(canvas, context, item)
    _indicator_summary(canvas, item)

    records = context.technical_history[item.symbol]
    _panel(canvas, 0.395, 0.225)
    _txt(canvas, 0.09, 0.602, "▥  Evolución de precios y medias móviles", color=INK, fontsize=14.5, fontweight="bold", va="center")
    price_axis = figure.add_axes([0.105, 0.425, 0.79, 0.145])
    _price_chart(price_axis, records)

    _panel(canvas, 0.278, 0.105)
    indicator_title = "RSI (14)" if indicator == "rsi" else "MACD (12,26,9)"
    _txt(canvas, 0.09, 0.357, f"◴  {indicator_title}", color=INK, fontsize=14.5, fontweight="bold", va="center")
    indicator_axis = figure.add_axes([0.105, 0.292, 0.79, 0.052])
    current = _rsi_chart(indicator_axis, records) if indicator == "rsi" else _macd_chart(indicator_axis, records)
    _txt(canvas, 0.89, 0.357, current, color=WHITE, fontsize=13, fontweight="bold", ha="right", va="center",
         bbox={"boxstyle": "round,pad=0.45", "facecolor": "#075AC4", "edgecolor": "none"})

    _analysis_box(canvas, item, sentiment, insight)
    _factors_box(canvas, item, sentiment, insight)
    _txt(canvas, 0.5, 0.008,
         f"Fuentes: RAG Morning Grain Comments + {context.technical_source_file} · No constituye recomendación de inversión.",
         color=MUTED, fontsize=7.8, ha="center")
    _save(figure, path)


def _hero(canvas, title, publication_date, page_number) -> None:
    canvas.add_patch(plt.Rectangle((0, 0.845), 1, 0.155, transform=canvas.transAxes, color=WHITE))
    canvas.add_patch(plt.Rectangle((0.045, 0.972), 0.067, 0.004, transform=canvas.transAxes, color="#E7B544"))
    _txt(canvas, 0.045, 0.95, "B O L E T Í N   D I A R I O", color=INK, fontsize=13, va="center")
    _txt(canvas, 0.045, 0.906, title, color=INK, fontsize=36, fontweight="bold", family="DejaVu Serif", va="center")
    canvas.add_patch(FancyBboxPatch((0.865, 0.955), 0.09, 0.03, boxstyle="round,pad=0.006,rounding_size=0.02",
                                    transform=canvas.transAxes, facecolor=TEAL, edgecolor="none"))
    _txt(canvas, 0.91, 0.97, f"{page_number}/5", color=WHITE, fontsize=13, fontweight="bold", ha="center", va="center")
    _txt(canvas, 0.05, 0.862, f"▣  {_display_date(publication_date)}", color=TEAL, fontsize=11.5, va="center")
    _txt(canvas, 0.33, 0.862, "●  Mercados agrícolas", color=TEAL, fontsize=11.5, va="center")
    _txt(canvas, 0.66, 0.862, "▥  Análisis diario", color=TEAL, fontsize=11.5, va="center")


def _instrument_header(canvas, symbol, name, market, sentiment) -> None:
    canvas.add_patch(FancyBboxPatch((0.03, 0.752), 0.94, 0.09, boxstyle="round,pad=0.006,rounding_size=0.022",
                                    transform=canvas.transAxes, facecolor=TEAL_DARK, edgecolor="none"))
    color = _sentiment_color(sentiment)
    canvas.add_patch(plt.Circle((0.092, 0.797), 0.034, transform=canvas.transAxes, fill=False, edgecolor=color, linewidth=2.4))
    _txt(canvas, 0.092, 0.797, symbol[0], color=color, fontsize=20, fontweight="bold", ha="center", va="center")
    _txt(canvas, 0.145, 0.808, f"{name} ({symbol})", color=WHITE, fontsize=21.5, fontweight="bold", va="center")
    _txt(canvas, 0.146, 0.778, market, color="#B9CAD4", fontsize=10.5, fontweight="bold", va="center")
    canvas.add_patch(FancyBboxPatch((0.74, 0.782), 0.2, 0.038, boxstyle="round,pad=0.007,rounding_size=0.025",
                                    transform=canvas.transAxes, facecolor=color, edgecolor="#FFFFFF66", linewidth=1.0))
    marker = "▲" if sentiment == "Alcista" else "▼" if sentiment == "Bajista" else "—"
    _txt(canvas, 0.84, 0.801, f"{marker}  {sentiment}", color=WHITE, fontsize=13.5, fontweight="bold", ha="center", va="center")


def _price_summary(canvas, context, item) -> None:
    _txt(canvas, 0.075, 0.725, "Precio de cierre", color=INK, fontsize=13)
    _txt(canvas, 0.07, 0.68, _number(item.price_close), color="#031C58", fontsize=38, fontweight="bold", va="center")
    canvas.add_line(Line2D([0.525, 0.525], [0.665, 0.735], transform=canvas.transAxes, color="#CDDCE2", linewidth=1))
    _txt(canvas, 0.58, 0.725, "Variación diaria", color=INK, fontsize=13)
    delta = item.variation_pct
    positive = delta is not None and delta >= 0
    color, arrow = (GREEN, "▲") if positive else (RED, "▼")
    _txt(canvas, 0.58, 0.688, f"{arrow} {_percent(delta)}", color=color, fontsize=27, fontweight="bold", va="center")
    _txt(canvas, 0.67, 0.662, f"({_signed_number(_absolute_change(context.technical_history[item.symbol]))})",
         color=color, fontsize=12.5, va="center")


def _indicator_summary(canvas, item) -> None:
    values = (("MA20", _number(item.ma20)), ("EMA50", _number(item.ema50)),
              ("RSI", _number(item.rsi, 1)), ("MACD", _signed_number(item.macd)))
    for index, (label, value) in enumerate(values):
        x = 0.055 + index * 0.232
        canvas.add_patch(FancyBboxPatch((x, 0.628), 0.212, 0.029, boxstyle="round,pad=0.004,rounding_size=0.012",
                                        transform=canvas.transAxes, facecolor="#EEF4F6", edgecolor=GRID, linewidth=0.8))
        _txt(canvas, x + 0.018, 0.6425, label, color=MUTED, fontsize=8.8, fontweight="bold", va="center")
        _txt(canvas, x + 0.194, 0.6425, value, color=INK, fontsize=10.5, fontweight="bold", ha="right", va="center")


def _panel(canvas, y, height) -> None:
    canvas.add_patch(FancyBboxPatch((0.045, y), 0.91, height, boxstyle="round,pad=0.006,rounding_size=0.02",
                                    transform=canvas.transAxes, facecolor=WHITE, edgecolor="#DFE9EC", linewidth=0.9))


def _price_chart(axis, records) -> None:
    dates = [_date(record.date) for record in records]
    close, ma20, ema50 = ([getattr(record, field) for record in records] for field in ("price_close", "ma20", "ema50"))
    axis.plot(dates, close, color=PRICE, linewidth=2.2, marker="o", markersize=2.2, label="Precio")
    if any(value is not None for value in ma20): axis.plot(dates, ma20, color=ORANGE, linewidth=1.7, label="MA20")
    if any(value is not None for value in ema50): axis.plot(dates, ema50, color=BLUE, linewidth=1.7, label="EMA50")
    axis.grid(color=GRID, linewidth=0.75, linestyle="--")
    axis.tick_params(axis="both", labelsize=9, colors=INK, length=0)
    axis.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=5))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    axis.legend(frameon=False, fontsize=9.5, ncol=3, loc="upper right")
    axis.margins(x=0.01)
    for spine in axis.spines.values(): spine.set_visible(False)


def _rsi_chart(axis, records) -> str:
    dates, values = [_date(record.date) for record in records], [record.rsi for record in records]
    axis.axhspan(30, 70, color=BLUE, alpha=0.045)
    axis.plot(dates, values, color="#304BFF", linewidth=1.8)
    for value in (70, 30): axis.axhline(value, color=GRID, linewidth=0.8, linestyle="--")
    axis.set_ylim(0, 100); axis.set_yticks([20, 40, 60, 80]); axis.set_xticks([])
    axis.grid(axis="y", color=GRID, linewidth=0.7, linestyle="--")
    axis.tick_params(axis="y", labelsize=8, colors=INK, length=0)
    for spine in axis.spines.values(): spine.set_visible(False)
    current = next((value for value in reversed(values) if value is not None), None)
    return "n.d." if current is None else f"{current:.0f}"


def _macd_chart(axis, records) -> str:
    dates = [_date(record.date) for record in records]
    macd, signal = [record.macd for record in records], [record.macd_signal for record in records]
    histogram = [left - right if left is not None and right is not None else 0.0 for left, right in zip(macd, signal)]
    axis.bar(dates, histogram, color=[GREEN if value >= 0 else RED for value in histogram], width=0.7, alpha=0.9, label="Histograma")
    axis.plot(dates, macd, color=BLUE, linewidth=1.3, label="MACD")
    axis.plot(dates, signal, color=ORANGE, linewidth=1.3, label="Señal")
    axis.axhline(0, color=MUTED, linewidth=0.65); axis.set_xticks([])
    axis.grid(axis="y", color=GRID, linewidth=0.7, linestyle="--")
    axis.tick_params(axis="y", labelsize=8, colors=INK, length=0)
    axis.legend(frameon=False, fontsize=7.5, ncol=3, loc="upper left")
    for spine in axis.spines.values(): spine.set_visible(False)
    return "n.d." if not histogram else f"{histogram[-1]:+.2f}"


def _analysis_box(canvas, item, sentiment, insight: CommodityInsight | None) -> None:
    color = _sentiment_color(sentiment)
    canvas.add_patch(FancyBboxPatch((0.045, 0.155), 0.91, 0.105, boxstyle="round,pad=0.006,rounding_size=0.02",
                                    transform=canvas.transAxes, facecolor=_soft_color(sentiment), edgecolor="none"))
    canvas.add_patch(plt.Rectangle((0.075, 0.172), 0.007, 0.052, transform=canvas.transAxes, color=color))
    _txt(canvas, 0.105, 0.231, "▥  Resumen y análisis técnico", color=INK, fontsize=15, fontweight="bold", va="center")
    lines = insight.analysis_lines if insight else _fallback_analysis(item)
    _txt(canvas, 0.108, 0.198, "\n".join(_fit(line, 94) for line in lines[:2]), color=INK, fontsize=11.7, va="center", linespacing=1.45)


def _factors_box(canvas, item, sentiment, insight: CommodityInsight | None) -> None:
    canvas.add_patch(FancyBboxPatch((0.045, 0.022), 0.91, 0.12, boxstyle="round,pad=0.006,rounding_size=0.02",
                                    transform=canvas.transAxes, facecolor=WHITE, edgecolor="#DFE9EC", linewidth=0.9))
    _txt(canvas, 0.075, 0.125, "▣  Factores clave y contexto", color=INK, fontsize=15, fontweight="bold", va="center")
    factors = insight.factors if insight else _fallback_factors(item, sentiment)
    for index, factor in enumerate(factors[:3], start=1):
        y = 0.102 - (index - 1) * 0.028
        _txt(canvas, 0.094, y, index, color=WHITE, fontsize=9, fontweight="bold", ha="center", va="center",
             bbox={"boxstyle": "circle,pad=0.36", "facecolor": AMBER, "edgecolor": "none"})
        _txt(canvas, 0.14, y, _fit(factor, 112), color=INK, fontsize=10.7, va="center")


def _fallback_analysis(item) -> tuple[str, str]:
    first = ("Precio por encima de MA20; el sesgo técnico permanece positivo." if item.price_close is not None and item.ma20 is not None and item.price_close >= item.ma20
             else "Precio por debajo de MA20; el sesgo técnico es defensivo.")
    direction = "por encima" if item.macd is not None and item.macd_signal is not None and item.macd >= item.macd_signal else "por debajo"
    return first, f"RSI en {_number(item.rsi, 0)}; MACD {direction} de su señal."


def _fallback_factors(item, sentiment) -> tuple[str, str, str]:
    relative = ("El cierre se mantiene por encima de la media de 20 sesiones." if item.price_close is not None and item.ma20 is not None and item.price_close >= item.ma20
                else "El cierre se mantiene por debajo de la media de 20 sesiones.")
    return relative, f"RSI actual de {_number(item.rsi, 1)} y MACD de {_number(item.macd)}.", f"Variación diaria de {_percent(item.variation_pct)}; lectura {sentiment.lower()}."


def _rule_sentiment(item) -> str:
    score = 0
    conditions = (None if item.variation_pct is None else item.variation_pct >= 0,
                  None if item.price_close is None or item.ma20 is None else item.price_close >= item.ma20,
                  None if item.rsi is None else item.rsi >= 50,
                  None if item.macd is None or item.macd_signal is None else item.macd >= item.macd_signal)
    for condition in conditions:
        if condition is not None: score += 1 if condition else -1
    return "Alcista" if score >= 2 else "Bajista" if score <= -2 else "Neutral"


def _fit(value: object, limit: int) -> str:
    text = " ".join(str(value).split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _absolute_change(records) -> float | None:
    closes = [record.price_close for record in records if record.price_close is not None]
    return None if len(closes) < 2 else closes[-1] - closes[-2]


def _sentiment_color(sentiment): return GREEN if sentiment == "Alcista" else RED if sentiment == "Bajista" else "#8B9AA2"
def _soft_color(sentiment): return "#EDF8F4" if sentiment == "Alcista" else "#FFF1F1" if sentiment == "Bajista" else "#FBF7ED"
def _date(value): return datetime.fromisoformat(value)
def _display_date(value): return datetime.fromisoformat(value).strftime("%d %b %Y")
def _number(value, decimals=2): return "n.d." if value is None else f"{value:,.{decimals}f}"
def _signed_number(value): return "n.d." if value is None else f"{value:+.2f}"
def _percent(value): return "n.d." if value is None else f"{value:+.2f}%"


def _save(figure, path) -> None:
    temporary = path.with_suffix(".tmp")
    figure.savefig(temporary, format="png", dpi=100, facecolor=figure.get_facecolor())
    plt.close(figure)
    temporary.replace(path)
