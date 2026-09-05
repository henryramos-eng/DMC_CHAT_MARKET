"""Construccion reproducible del indice RAG para multiples reportes PDF."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from .chunking import chunk_pages
from .cleaning import clean_pages
from .config import RAGConfig
from .dates import publication_date_from_filename
from .extraction import extract_pages, validate_pdf
from .openai_client import OpenAIEmbedder
from .vector_index import LocalVectorIndex


def discover_reports(pdf_dir: Path, pattern: str) -> list[tuple[str, Path]]:
    if not pdf_dir.is_dir():
        raise FileNotFoundError(f"No existe el directorio de PDFs: {pdf_dir.resolve()}")

    reports: list[tuple[str, Path]] = []
    dates_seen: dict[str, Path] = {}
    for path in pdf_dir.glob(pattern):
        if not path.is_file():
            continue
        publication_date = publication_date_from_filename(path)
        if publication_date in dates_seen:
            raise ValueError(
                "Hay dos reportes para la misma fecha: "
                f"{dates_seen[publication_date].name} y {path.name}"
            )
        dates_seen[publication_date] = path
        reports.append((publication_date, path))

    if not reports:
        raise FileNotFoundError(
            f"No se encontraron PDFs en {pdf_dir.resolve()} con el patron {pattern}"
        )
    return sorted(reports, key=lambda item: (item[0], item[1].name.casefold()))


def build_index(
    config: RAGConfig,
    *,
    extract_only: bool = False,
    force: bool = False,
) -> dict[str, object]:
    config.validate_chunking()
    report_paths = discover_reports(config.pdf_dir, config.pdf_pattern)

    all_raw_pages: list[dict[str, object]] = []
    all_clean_pages: list[dict[str, object]] = []
    all_chunks = []
    documents: list[dict[str, object]] = []

    for publication_date, path in report_paths:
        source = validate_pdf(path)
        raw_pages = extract_pages(source)
        cleaned_pages, cleaning_report = clean_pages(raw_pages)
        chunks = chunk_pages(
            cleaned_pages,
            source.filename,
            publication_date=publication_date,
            chunk_size_tokens=config.chunk_size_tokens,
            overlap_tokens=config.chunk_overlap_tokens,
        )
        if not chunks:
            raise ValueError(
                f"La limpieza no produjo contenido para indexar en {source.filename}"
            )

        all_raw_pages.extend(
            {
                "source_file": source.filename,
                "publication_date": publication_date,
                "page_number": page.page_number,
                "raw_text": page.raw_text,
            }
            for page in raw_pages
        )
        all_clean_pages.extend(
            {
                "source_file": source.filename,
                "publication_date": publication_date,
                "page_number": page.page_number,
                "clean_text": page.clean_text,
            }
            for page in cleaned_pages
        )
        all_chunks.extend(chunks)
        documents.append(
            {
                "source_file": source.filename,
                "source_path": str(source.path),
                "publication_date": publication_date,
                "source_sha256": source.sha256,
                "source_size_bytes": source.size_bytes,
                "page_count": source.page_count,
                "chunk_count": len(chunks),
                "cleaning": cleaning_report.to_dict(),
            }
        )

    config.index_dir.mkdir(parents=True, exist_ok=True)
    _write_json(config.index_dir / "raw_pages.json", all_raw_pages)
    _write_json(config.index_dir / "clean_pages.json", all_clean_pages)
    _write_json(
        config.index_dir / "chunks.json",
        [chunk.to_dict() for chunk in all_chunks],
    )

    source_set_sha256 = _source_set_hash(documents)
    manifest: dict[str, object] = {
        "document_count": len(documents),
        "available_dates": [item[0] for item in report_paths],
        "documents": documents,
        "source_set_sha256": source_set_sha256,
        "page_count": sum(int(item["page_count"]) for item in documents),
        "chunk_count": len(all_chunks),
        "chunk_size_tokens": config.chunk_size_tokens,
        "chunk_overlap_tokens": config.chunk_overlap_tokens,
        "embedding_model": config.embedding_model,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "vectors_created": False,
    }

    if extract_only:
        _write_json(config.index_dir / "quality_report.json", manifest)
        return manifest

    if not config.openai_api_key:
        raise ValueError(
            "Falta OPENAI_API_KEY. Usa --extract-only o configura la clave en .env"
        )
    if not force and _same_complete_index(config.index_dir, manifest):
        return json.loads(
            (config.index_dir / "quality_report.json").read_text(encoding="utf-8")
        )

    embedder = OpenAIEmbedder(config.openai_api_key, config.embedding_model)
    vectors = embedder.embed_texts([chunk.text for chunk in all_chunks])
    LocalVectorIndex(vectors, all_chunks).save(config.index_dir)
    manifest["vectors_created"] = True
    manifest["vector_dimensions"] = int(vectors.shape[1])
    _write_json(config.index_dir / "quality_report.json", manifest)
    return manifest


def _source_set_hash(documents: list[dict[str, object]]) -> str:
    digest = hashlib.sha256()
    for document in documents:
        digest.update(
            (
                f"{document['publication_date']}:{document['source_file']}:"
                f"{document['source_sha256']}\n"
            ).encode("utf-8")
        )
    return digest.hexdigest()


def _same_complete_index(directory: Path, expected: dict[str, object]) -> bool:
    manifest_path = directory / "quality_report.json"
    if not manifest_path.is_file():
        return False
    current = json.loads(manifest_path.read_text(encoding="utf-8"))
    keys = (
        "source_set_sha256",
        "chunk_size_tokens",
        "chunk_overlap_tokens",
        "embedding_model",
    )
    return (
        current.get("vectors_created") is True
        and all(current.get(key) == expected.get(key) for key in keys)
        and (directory / "vectors.npz").is_file()
        and (directory / "metadata.json").is_file()
    )


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extrae, limpia, fragmenta e indexa reportes PDF por fecha."
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        help="Un unico PDF; conserva compatibilidad con la version anterior",
    )
    parser.add_argument("--pdf-dir", type=Path, help="Directorio de reportes PDF")
    parser.add_argument("--pattern", help="Patron glob de nombres de reporte")
    parser.add_argument(
        "--index-dir", type=Path, help="Directorio del indice; reemplaza RAG_INDEX_DIR"
    )
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = RAGConfig.from_env()
    if args.pdf:
        config = replace(
            config,
            pdf_dir=args.pdf.parent,
            pdf_pattern=args.pdf.name,
        )
    if args.pdf_dir:
        config = replace(config, pdf_dir=args.pdf_dir)
    if args.pattern:
        config = replace(config, pdf_pattern=args.pattern)
    if args.index_dir:
        config = replace(config, index_dir=args.index_dir)
    report = build_index(config, extract_only=args.extract_only, force=args.force)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
