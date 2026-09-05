"""Entrada desde la raiz para evaluar las preguntas frecuentes."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from rag.faq_evaluation import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
