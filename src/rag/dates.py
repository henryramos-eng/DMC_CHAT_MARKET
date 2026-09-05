"""Fechas de publicacion en nombres de archivo y consultas del usuario."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path


REPORT_FILENAME = re.compile(
    r"^morning_grain_comments_(\d{8})\.pdf$",
    re.IGNORECASE,
)
COMPACT_DATE = re.compile(r"(?<!\d)(\d{2})(\d{2})(\d{4})(?!\d)")
SEPARATED_DATE = re.compile(
    r"(?<!\d)(\d{1,2})[/-](\d{1,2})[/-](\d{4})(?!\d)"
)
ISO_DATE = re.compile(r"(?<!\d)(\d{4})-(\d{1,2})-(\d{1,2})(?!\d)")
SPANISH_DATE = re.compile(
    r"(?<!\d)(\d{1,2})\s+de\s+"
    r"(enero|febrero|marzo|abril|mayo|junio|julio|agosto|"
    r"septiembre|setiembre|octubre|noviembre|diciembre)"
    r"\s+de\s+(\d{4})(?!\d)",
    re.IGNORECASE,
)

MONTHS = {
    "enero": 1,
    "febrero": 2,
    "marzo": 3,
    "abril": 4,
    "mayo": 5,
    "junio": 6,
    "julio": 7,
    "agosto": 8,
    "septiembre": 9,
    "setiembre": 9,
    "octubre": 10,
    "noviembre": 11,
    "diciembre": 12,
}


class DateQueryError(ValueError):
    """La consulta parece contener una fecha invalida o ambigua."""


@dataclass(frozen=True)
class ParsedQueryDate:
    publication_date: str | None
    semantic_query: str


def publication_date_from_filename(path: str | Path) -> str:
    filename = Path(path).name
    match = REPORT_FILENAME.fullmatch(filename)
    if not match:
        raise ValueError(
            "Nombre de reporte invalido: "
            f"{filename}. Se esperaba morning_grain_comments_DDMMYYYY.pdf"
        )
    try:
        return datetime.strptime(match.group(1), "%d%m%Y").date().isoformat()
    except ValueError as exc:
        raise ValueError(f"Fecha invalida en el archivo {filename}") from exc


def parse_query_date(text: str) -> ParsedQueryDate:
    matches: list[tuple[tuple[int, int], date]] = []
    occupied: list[tuple[int, int]] = []

    for pattern, converter in (
        (ISO_DATE, _from_iso_match),
        (SEPARATED_DATE, _from_day_month_match),
        (COMPACT_DATE, _from_day_month_match),
        (SPANISH_DATE, _from_spanish_match),
    ):
        for match in pattern.finditer(text):
            span = match.span()
            if any(_overlaps(span, previous) for previous in occupied):
                continue
            try:
                parsed = converter(match)
            except ValueError as exc:
                raise DateQueryError(
                    f"La fecha '{match.group(0)}' no es valida."
                ) from exc
            occupied.append(span)
            matches.append((span, parsed))

    unique_dates = {item.isoformat() for _, item in matches}
    if len(unique_dates) > 1:
        raise DateQueryError("Indica una sola fecha de publicacion por consulta.")

    semantic = text
    for (start, end), _ in sorted(matches, reverse=True):
        semantic = f"{semantic[:start]} {semantic[end:]}"
    semantic = re.sub(
        r"\b(?:para\s+la\s+fecha|en\s+la\s+fecha|para\s+fecha|fecha)\b",
        " ",
        semantic,
        flags=re.IGNORECASE,
    )
    semantic = re.sub(r"\s+", " ", semantic).strip(" ,.;:-")
    publication_date = next(iter(unique_dates), None)
    return ParsedQueryDate(publication_date, semantic)


def is_available_dates_question(text: str) -> bool:
    normalized = _normalize(text)
    phrases = (
        "que fechas hay",
        "que fechas tienes",
        "fechas disponibles",
        "reportes disponibles",
        "que reportes hay",
        "que reportes tienes",
        "de que fechas",
    )
    return any(phrase in normalized for phrase in phrases)


def requests_latest_report(text: str) -> bool:
    normalized = _normalize(text)
    terms = (
        "hoy",
        "actual",
        "ultimo",
        "ultima",
        "mas reciente",
        "reciente",
    )
    return any(re.search(rf"\b{re.escape(term)}\b", normalized) for term in terms)


def display_date(value: str) -> str:
    return datetime.strptime(value, "%Y-%m-%d").strftime("%d/%m/%Y")


def compact_date(value: str) -> str:
    return datetime.strptime(value, "%Y-%m-%d").strftime("%d%m%Y")


def _from_iso_match(match: re.Match[str]) -> date:
    return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))


def _from_day_month_match(match: re.Match[str]) -> date:
    return date(int(match.group(3)), int(match.group(2)), int(match.group(1)))


def _from_spanish_match(match: re.Match[str]) -> date:
    month = MONTHS[match.group(2).casefold()]
    return date(int(match.group(3)), month, int(match.group(1)))


def _overlaps(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def _normalize(text: str) -> str:
    value = unicodedata.normalize("NFKD", text.casefold())
    value = "".join(char for char in value if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", value).strip()
