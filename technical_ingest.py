"""Convierte el Excel tecnico a JSON sin incorporarlo al RAG."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from technical.excel_loader import load_technical_excel  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Valida y convierte el Excel tecnico")
    parser.add_argument(
        "--excel",
        type=Path,
        default=Path("data/technical/checklist_tecnico.xlsx"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/technical/checklist_tecnico.json"),
    )
    args = parser.parse_args()
    dataset = load_technical_excel(args.excel)
    dataset.save_json(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "records": len(dataset.records),
                "dates": len(dataset.dates),
                "symbols": list(dataset.symbols),
                "warnings": list(dataset.warnings),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
