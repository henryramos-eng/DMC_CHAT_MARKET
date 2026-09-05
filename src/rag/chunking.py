"""Chunking por pagina con ventanas de tokens solapadas."""

from __future__ import annotations

import hashlib
from pathlib import Path

import tiktoken

from .models import Chunk, ExtractedPage


def chunk_pages(
    pages: list[ExtractedPage],
    source_file: str | Path,
    *,
    publication_date: str = "",
    chunk_size_tokens: int = 700,
    overlap_tokens: int = 120,
) -> list[Chunk]:
    if chunk_size_tokens <= 0:
        raise ValueError("chunk_size_tokens debe ser mayor que cero")
    if not 0 <= overlap_tokens < chunk_size_tokens:
        raise ValueError("overlap_tokens debe ser menor que chunk_size_tokens")

    encoder = tiktoken.get_encoding("cl100k_base")
    step = chunk_size_tokens - overlap_tokens
    filename = Path(source_file).name
    chunks: list[Chunk] = []

    for page in pages:
        if not page.clean_text.strip():
            continue
        tokens = encoder.encode(page.clean_text)
        for start in range(0, len(tokens), step):
            end = min(start + chunk_size_tokens, len(tokens))
            text = encoder.decode(tokens[start:end]).strip()
            if not text:
                continue
            fingerprint = hashlib.sha1(
                (
                    f"{filename}:{publication_date}:{page.page_number}:"
                    f"{start}:{end}:{text}"
                ).encode("utf-8")
            ).hexdigest()[:12]
            chunks.append(
                Chunk(
                    chunk_id=f"p{page.page_number}-{fingerprint}",
                    text=text,
                    page_number=page.page_number,
                    source_file=filename,
                    start_token=start,
                    end_token=end,
                    publication_date=publication_date,
                )
            )
            if end == len(tokens):
                break
    return chunks
