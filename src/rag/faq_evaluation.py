"""Ejecuta el FAQ como conjunto de evaluacion, nunca como fuente del RAG."""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from .config import RAGConfig
from .dates import compact_date, parse_query_date
from .service import RAGService


async def evaluate_faq(
    config: RAGConfig,
    faq_path: Path,
    output_path: Path,
) -> list[dict[str, object]]:
    questions = json.loads(faq_path.read_text(encoding="utf-8"))
    service = RAGService(config)
    results: list[dict[str, object]] = []
    for index, item in enumerate(questions, start=1):
        question = item["question"]
        publication_date = item.get("publication_date")
        if publication_date and not parse_query_date(question).publication_date:
            evaluated_question = f"{question} {compact_date(publication_date)}"
        else:
            evaluated_question = question
        answer = await service.handle(10_000 + index, evaluated_question)
        results.append(
            {
                **item,
                "evaluated_question": evaluated_question,
                "answer": answer,
                "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        print(f"[{index}/{len(questions)}] {evaluated_question}\n{answer}\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evalua las preguntas FAQ contra el RAG.")
    parser.add_argument(
        "--faq",
        type=Path,
        default=Path("data/faq/morning_grain_comments_04092026.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/reports/rag/faq_results.json"),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    asyncio.run(evaluate_faq(RAGConfig.from_env(), args.faq, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
