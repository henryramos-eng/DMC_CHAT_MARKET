"""Limpieza conservadora y trazable del texto extraido."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from .models import ExtractedPage


EMAIL_LINE = re.compile(r"(?im)^\s*[\w.+-]+@[\w.-]+\.[a-z]{2,}\s*$")
BULLET_REPLACEMENT = re.compile(r"(?m)^\s*\ufffd\s+")
HYPHENATED_LINE = re.compile(r"(?<=\w)-\s*\n\s*(?=\w)")
MULTIPLE_SPACES = re.compile(r"[ \t]{2,}")
MULTIPLE_BLANKS = re.compile(r"\n{3,}")


@dataclass(frozen=True)
class CleaningReport:
    total_pages: int
    pages_with_raw_text: int
    pages_with_clean_text: int
    empty_pages: tuple[int, ...]
    raw_characters: int
    clean_characters: int
    replacement_characters_before: int
    replacement_characters_after: int

    def to_dict(self) -> dict[str, object]:
        return {
            "total_pages": self.total_pages,
            "pages_with_raw_text": self.pages_with_raw_text,
            "pages_with_clean_text": self.pages_with_clean_text,
            "empty_pages": list(self.empty_pages),
            "raw_characters": self.raw_characters,
            "clean_characters": self.clean_characters,
            "replacement_characters_before": self.replacement_characters_before,
            "replacement_characters_after": self.replacement_characters_after,
        }


def clean_pages(pages: list[ExtractedPage]) -> tuple[list[ExtractedPage], CleaningReport]:
    cleaned: list[ExtractedPage] = []
    for page in pages:
        cleaned.append(
            ExtractedPage(
                page_number=page.page_number,
                raw_text=page.raw_text,
                clean_text=clean_text(page.raw_text),
            )
        )

    report = CleaningReport(
        total_pages=len(pages),
        pages_with_raw_text=sum(bool(page.raw_text.strip()) for page in pages),
        pages_with_clean_text=sum(bool(page.clean_text.strip()) for page in cleaned),
        empty_pages=tuple(
            page.page_number for page in cleaned if not page.clean_text.strip()
        ),
        raw_characters=sum(len(page.raw_text) for page in pages),
        clean_characters=sum(len(page.clean_text) for page in cleaned),
        replacement_characters_before=sum(page.raw_text.count("\ufffd") for page in pages),
        replacement_characters_after=sum(page.clean_text.count("\ufffd") for page in cleaned),
    )
    return cleaned, report


def clean_text(raw_text: str) -> str:
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    text = unicodedata.normalize("NFKC", text)
    text = BULLET_REPLACEMENT.sub("- ", text)
    text = text.replace("\ufffds", "'s")
    text = re.sub(r"\ufffd(?=\d{2}/\d{2})", "'", text)
    text = re.sub(r"\(\ufffd([^\n()]+?)\ufffd\)", r'(\1)', text)
    text = text.replace("\ufffd", "'")
    text = HYPHENATED_LINE.sub("", text)
    text = EMAIL_LINE.sub("", text)

    normalized_lines = [MULTIPLE_SPACES.sub(" ", line).strip() for line in text.splitlines()]
    text = "\n".join(normalized_lines)
    text = MULTIPLE_BLANKS.sub("\n\n", text)
    return text.strip()
