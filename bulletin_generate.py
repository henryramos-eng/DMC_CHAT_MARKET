"""Genera localmente un boletin PNG sin iniciar Telegram."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from bulletin.config import BulletinConfig  # noqa: E402
from bulletin.pipeline import BulletinPipeline  # noqa: E402
from rag.config import RAGConfig  # noqa: E402
from rag.dates import parse_query_date  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Genera el boletin como imagenes PNG")
    parser.add_argument("--date", help="DDMMYYYY, DD/MM/YYYY o YYYY-MM-DD")
    parser.add_argument("--symbols", help="Simbolos separados por comas")
    args = parser.parse_args()
    publication_date = None
    if args.date:
        publication_date = parse_query_date(args.date).publication_date
        if publication_date is None:
            raise ValueError("No se pudo interpretar --date")
    symbols = (
        tuple(item.strip().upper() for item in args.symbols.split(",") if item.strip())
        if args.symbols
        else None
    )
    result = BulletinPipeline(
        BulletinConfig.from_env(),
        RAGConfig.from_env(),
    ).run(publication_date, symbols=symbols)
    print(
        json.dumps(
            {
                "publication_date": result.publication_date,
                "images": [str(path) for path in result.images],
                "manifest": str(result.manifest_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
