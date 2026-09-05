"""Intenciones conversacionales que no deben ejecutar el RAG."""

from __future__ import annotations

import re
import unicodedata
from enum import Enum


class Intent(str, Enum):
    GREETING = "greeting"
    THANKS = "thanks"
    GOODBYE = "goodbye"
    OUT_OF_SCOPE = "out_of_scope"
    REPORT_QUERY = "report_query"


GREETING = re.compile(r"^(hola|buenos dias|buenas tardes|buenas noches)[!. ]*$")
THANKS = re.compile(r"^(gracias|muchas gracias|te agradezco)[!. ]*$")
GOODBYE = re.compile(r"^(adios|hasta luego|nos vemos)[!. ]*$")
OBVIOUSLY_UNRELATED = (
    "mundial de futbol",
    "partido de futbol",
    "receta de cocina",
    "pelicula",
    "serie de television",
    "capital de",
    "programar en",
    "llanta",
    "llantas",
    "neumatico",
    "neumaticos",
)
UNRELATED_PATTERNS = (
    re.compile(r"\b(carro|automovil|motocicleta|camioneta)\b.*\b(llanta|rueda)"),
    re.compile(r"\b(llanta|rueda)s?\b.*\b(carro|automovil|vehiculo)\b"),
)


def classify_intent(text: str) -> Intent:
    normalized = normalize(text)
    if GREETING.fullmatch(normalized):
        return Intent.GREETING
    if THANKS.fullmatch(normalized):
        return Intent.THANKS
    if GOODBYE.fullmatch(normalized):
        return Intent.GOODBYE
    if any(term in normalized for term in OBVIOUSLY_UNRELATED):
        return Intent.OUT_OF_SCOPE
    if any(pattern.search(normalized) for pattern in UNRELATED_PATTERNS):
        return Intent.OUT_OF_SCOPE
    return Intent.REPORT_QUERY


def normalize(text: str) -> str:
    value = unicodedata.normalize("NFKD", text.casefold())
    value = "".join(char for char in value if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", value).strip()
