"""Prueba aislada de inferencia del adaptador LoRA entrenado."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bulletin.lora_generator import LoRABulletinGenerator  # noqa: E402
from bulletin.models import UnifiedBulletinContext  # noqa: E402
from technical.models import TechnicalRecord  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Ejecuta inferencia con base + LoRA")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--context", required=True, type=Path)
    args = parser.parse_args()
    raw = json.loads(args.context.read_text(encoding="utf-8"))
    technical = raw["technical_context"]
    context = UnifiedBulletinContext(
        publication_date=raw["publication_date"],
        rag_chunks=tuple(raw["rag_context"]),
        technical_snapshot=tuple(
            TechnicalRecord.from_dict(item) for item in technical["snapshot"]
        ),
        technical_history={
            symbol: tuple(TechnicalRecord.from_dict(item) for item in values)
            for symbol, values in technical["history"].items()
        },
        technical_source_file=technical["source_file"],
        technical_warnings=tuple(technical.get("warnings", [])),
    )
    draft = LoRABulletinGenerator(args.base_model, args.adapter).generate(context)
    print(json.dumps(draft.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
