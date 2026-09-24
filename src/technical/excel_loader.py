"""Lectura validada de Excel tecnico y normalizacion a JSON."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .models import TechnicalDataset, TechnicalRecord


SUMMARY_COLUMNS = {
    "Commodity", "Fecha", "Senal", "Condiciones_Largo", "Condiciones_Corto"
}
DETAIL_COLUMNS = {
    "Commodity", "Fecha", "Indicador", "Valor", "Condicion Evaluada", "Resultado"
}
CANONICAL_COLUMNS = {
    "fecha", "simbolo", "precio_cierre", "variacion_pct", "ma20", "ma50",
    "rsi", "macd", "soporte", "resistencia", "tendencia",
}

NUMBER = r"[-+]?\d+(?:\.\d+)?"
EMA50 = re.compile(rf"EMA50=({NUMBER})", re.IGNORECASE)
EMA200 = re.compile(rf"EMA200=({NUMBER})", re.IGNORECASE)
CLOSE = re.compile(rf"Close=({NUMBER})", re.IGNORECASE)
MACD = re.compile(rf"^\s*({NUMBER})\s*/\s*({NUMBER})\s*$")

TREND_MAP = {
    "largo": "Alcista", "alcista": "Alcista",
    "corto": "Bajista", "bajista": "Bajista",
    "mixto": "Neutral", "neutral": "Neutral",
    "sin senal": "Neutral", "sin señal": "Neutral",
}


class TechnicalDataError(ValueError):
    """El Excel tecnico no cumple el contrato requerido."""


def load_technical_excel(path: Path) -> TechnicalDataset:
    source = path.resolve()
    _validate_source(source)
    try:
        workbook = pd.ExcelFile(source)
    except Exception as exc:
        raise TechnicalDataError(f"No se pudo abrir el Excel: {exc}") from exc

    if {"Checklist_Resumen", "Checklist_Detalle"}.issubset(workbook.sheet_names):
        summary = pd.read_excel(source, sheet_name="Checklist_Resumen")
        detail = pd.read_excel(source, sheet_name="Checklist_Detalle")
        return normalize_checklist(
            summary, detail, source_file=source.name, source_sha256=_sha256(source)
        )

    first_sheet = pd.read_excel(source, sheet_name=workbook.sheet_names[0])
    normalized_columns = {
        str(column).strip().casefold() for column in first_sheet.columns
    }
    if CANONICAL_COLUMNS.issubset(normalized_columns):
        return normalize_canonical(
            first_sheet, source_file=source.name, source_sha256=_sha256(source)
        )
    raise TechnicalDataError(
        "El libro debe contener Checklist_Resumen y Checklist_Detalle, "
        "o una hoja con el esquema tecnico canonico."
    )


def normalize_checklist(
    summary: pd.DataFrame,
    detail: pd.DataFrame,
    *,
    source_file: str,
    source_sha256: str,
) -> TechnicalDataset:
    _require_columns(summary, SUMMARY_COLUMNS, "Checklist_Resumen")
    _require_columns(detail, DETAIL_COLUMNS, "Checklist_Detalle")
    if summary.empty or detail.empty:
        raise TechnicalDataError("Las hojas tecnicas no pueden estar vacias")

    summary = summary.copy()
    detail = detail.copy()
    for frame in (summary, detail):
        frame["Commodity"] = frame["Commodity"].astype("string").str.strip().str.upper()
        frame["Fecha"] = _parse_dates(frame["Fecha"])

    _reject_nulls(summary, ["Commodity", "Fecha", "Senal"], "Checklist_Resumen")
    _reject_nulls(detail, ["Commodity", "Fecha", "Indicador", "Valor"], "Checklist_Detalle")
    _reject_duplicates(summary, ["Commodity", "Fecha"], "Checklist_Resumen")
    _reject_duplicates(detail, ["Commodity", "Fecha", "Indicador"], "Checklist_Detalle")

    wide = detail.pivot(
        index=["Commodity", "Fecha"], columns="Indicador", values="Valor"
    ).reset_index()
    missing_indicators = {"EMA", "MACD", "RSI"}.difference(wide.columns)
    if missing_indicators:
        raise TechnicalDataError(
            "Faltan indicadores requeridos: " + ", ".join(sorted(missing_indicators))
        )

    merged = summary.merge(
        wide, on=["Commodity", "Fecha"], how="left", validate="one_to_one"
    )
    merged["price_close"] = merged["EMA"].map(lambda value: _extract(value, CLOSE))
    merged["ema50"] = merged["EMA"].map(lambda value: _extract(value, EMA50))
    merged["ema200"] = merged["EMA"].map(lambda value: _extract(value, EMA200))
    # El libro puede anotar el RSI como "56.06 (prev: 72.88)". El primer
    # numero es el valor de la fecha; el segundo es solo una referencia previa.
    merged["rsi"] = merged["RSI"].map(_extract_first_number)
    macd_pairs = merged["MACD"].map(_extract_macd)
    merged["macd"] = macd_pairs.map(lambda pair: pair[0] if pair else None)
    merged["macd_signal"] = macd_pairs.map(lambda pair: pair[1] if pair else None)

    parsed_columns = ["price_close", "ema50", "ema200", "rsi", "macd", "macd_signal"]
    failures = {column: int(merged[column].isna().sum()) for column in parsed_columns}
    failures = {column: count for column, count in failures.items() if count}
    if failures:
        raise TechnicalDataError(f"No se pudieron interpretar indicadores: {failures}")

    merged = merged.sort_values(["Commodity", "Fecha"]).reset_index(drop=True)
    grouped_close = merged.groupby("Commodity", sort=False)["price_close"]
    merged["variation_pct"] = grouped_close.pct_change(fill_method=None) * 100.0
    merged["ma20"] = grouped_close.transform(
        lambda values: values.rolling(20, min_periods=20).mean()
    )
    merged["ma50"] = grouped_close.transform(
        lambda values: values.rolling(50, min_periods=50).mean()
    )
    merged["trend"] = merged["Senal"].map(_map_trend)

    records = tuple(
        TechnicalRecord(
            date=row.Fecha.date().isoformat(),
            symbol=str(row.Commodity),
            price_close=_optional_float(row.price_close),
            variation_pct=_optional_float(row.variation_pct),
            ma20=_optional_float(row.ma20),
            ma50=_optional_float(row.ma50),
            rsi=_optional_float(row.rsi),
            macd=_optional_float(row.macd),
            macd_signal=_optional_float(row.macd_signal),
            support=None,
            resistance=None,
            trend=str(row.trend),
            ema50=_optional_float(row.ema50),
            ema200=_optional_float(row.ema200),
            raw_signal=str(row.Senal),
        )
        for row in merged.itertuples(index=False)
    )
    return TechnicalDataset(
        schema_version="1.0",
        source_file=source_file,
        source_sha256=source_sha256,
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        records=records,
        warnings=(
            "El Excel no contiene soporte ni resistencia; se exportan como null.",
            "MA20 y MA50 se calculan con cierres historicos sin usar fechas futuras.",
            "EMA50 y EMA200 originales se preservan en campos separados.",
            "Cuando RSI contiene '(prev: ...)', se usa el primer valor como RSI actual.",
        ),
    )


def normalize_canonical(
    frame: pd.DataFrame,
    *,
    source_file: str,
    source_sha256: str,
) -> TechnicalDataset:
    data = frame.rename(
        columns={column: str(column).strip().casefold() for column in frame.columns}
    ).copy()
    _require_columns(data, CANONICAL_COLUMNS, "hoja canonica")
    data["fecha"] = _parse_dates(data["fecha"])
    data["simbolo"] = data["simbolo"].astype("string").str.strip().str.upper()
    _reject_nulls(data, ["fecha", "simbolo", "precio_cierre", "tendencia"], "hoja canonica")
    _reject_duplicates(data, ["simbolo", "fecha"], "hoja canonica")
    numeric = [
        "precio_cierre", "variacion_pct", "ma20", "ma50", "rsi", "macd",
        "soporte", "resistencia",
    ]
    for column in numeric:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    records = tuple(
        TechnicalRecord(
            date=row.fecha.date().isoformat(),
            symbol=str(row.simbolo),
            price_close=_optional_float(row.precio_cierre),
            variation_pct=_optional_float(row.variacion_pct),
            ma20=_optional_float(row.ma20),
            ma50=_optional_float(row.ma50),
            rsi=_optional_float(row.rsi),
            macd=_optional_float(row.macd),
            macd_signal=None,
            support=_optional_float(row.soporte),
            resistance=_optional_float(row.resistencia),
            trend=_map_trend(row.tendencia),
            raw_signal=str(row.tendencia),
        )
        for row in data.sort_values(["simbolo", "fecha"]).itertuples(index=False)
    )
    return TechnicalDataset(
        schema_version="1.0",
        source_file=source_file,
        source_sha256=source_sha256,
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        records=records,
    )


def _validate_source(path: Path) -> None:
    if not path.is_file():
        raise TechnicalDataError(f"No existe el Excel tecnico: {path}")
    if path.stat().st_size == 0:
        raise TechnicalDataError(f"El Excel tecnico esta vacio: {path}")
    with path.open("rb") as stream:
        signature = stream.read(4)
    if signature != b"PK\x03\x04":
        raise TechnicalDataError("El archivo no tiene una firma XLSX valida")


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = required.difference(frame.columns)
    if missing:
        raise TechnicalDataError(
            f"{label}: faltan columnas requeridas: {', '.join(sorted(missing))}"
        )


def _reject_nulls(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    counts = frame[columns].isna().sum()
    failures = {column: int(count) for column, count in counts.items() if count}
    if failures:
        raise TechnicalDataError(f"{label}: valores requeridos ausentes: {failures}")


def _reject_duplicates(frame: pd.DataFrame, key: list[str], label: str) -> None:
    count = int(frame.duplicated(key, keep=False).sum())
    if count:
        raise TechnicalDataError(f"{label}: {count} filas duplicadas para la clave {key}")


def _parse_dates(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce")
    if parsed.isna().any():
        raise TechnicalDataError(
            f"Hay {int(parsed.isna().sum())} fechas invalidas en el Excel"
        )
    return parsed.dt.normalize()


def _extract(value: object, pattern: re.Pattern[str]) -> float | None:
    match = pattern.search(str(value))
    return float(match.group(1)) if match else None


def _extract_first_number(value: object) -> float | None:
    match = re.search(NUMBER, str(value))
    return float(match.group(0)) if match else None


def _extract_macd(value: object) -> tuple[float, float] | None:
    match = MACD.fullmatch(str(value))
    if not match:
        return None
    return float(match.group(1)), float(match.group(2))


def _map_trend(value: object) -> str:
    normalized = str(value).strip().casefold()
    if normalized not in TREND_MAP:
        raise TechnicalDataError(f"Tendencia no reconocida: {value}")
    return TREND_MAP[normalized]


def _optional_float(value: object) -> float | None:
    if pd.isna(value):
        return None
    return round(float(value), 6)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
