"""Validacion y extraccion fiel del PDF, sin transformaciones de negocio."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

from .models import ExtractedPage


LOGGER = logging.getLogger(__name__)


class PDFValidationError(ValueError):
    """El archivo no puede utilizarse como fuente confiable."""


@dataclass(frozen=True)
class PDFSource:
    path: Path
    filename: str
    sha256: str
    size_bytes: int
    page_count: int


def validate_pdf(path: Path) -> PDFSource:
    resolved = path.resolve()
    if not resolved.is_file():
        raise PDFValidationError(f"No existe el PDF: {resolved}")
    size = resolved.stat().st_size
    if size == 0:
        raise PDFValidationError(f"El PDF esta vacio: {resolved}")
    with resolved.open("rb") as stream:
        signature = stream.read(5)
    if signature != b"%PDF-":
        raise PDFValidationError(f"El archivo no tiene una firma PDF valida: {resolved}")

    reader = _open_reader(resolved)
    if not reader.pages:
        raise PDFValidationError("El PDF no contiene paginas")

    return PDFSource(
        path=resolved,
        filename=resolved.name,
        sha256=_sha256(resolved),
        size_bytes=size,
        page_count=len(reader.pages),
    )


def extract_pages(source: PDFSource) -> list[ExtractedPage]:
    reader = _open_reader(source.path)
    pages: list[ExtractedPage] = []
    for index, page in enumerate(reader.pages, start=1):
        try:
            raw_text = page.extract_text() or ""
        except Exception as exc:
            raise PDFValidationError(
                f"No se pudo extraer la pagina {index}: {exc}"
            ) from exc
        pages.append(ExtractedPage(page_number=index, raw_text=raw_text))

    if not any(page.raw_text.strip() for page in pages):
        raise PDFValidationError(
            "El PDF no contiene texto extraible; esta version requiere OCR"
        )
    return pages


def _open_reader(path: Path) -> PdfReader:
    previous_level = logging.getLogger("pypdf").level
    logging.getLogger("pypdf").setLevel(logging.ERROR)
    try:
        return PdfReader(str(path), strict=False)
    except Exception as exc:
        raise PDFValidationError(f"No se pudo abrir el PDF: {exc}") from exc
    finally:
        logging.getLogger("pypdf").setLevel(previous_level)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
