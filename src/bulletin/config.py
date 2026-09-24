"""Configuración del pipeline de boletines, separada del RAG."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class BulletinConfig:
    technical_excel_path: Path
    technical_json_path: Path
    output_dir: Path
    text_backend: str = "openai"
    openai_model: str = ""
    openai_timeout_seconds: float = 60.0
    llm_fallback_enabled: bool = True
    lora_base_model: str = ""
    lora_adapter_path: Path | None = None
    history_days: int = 60

    @classmethod
    def from_env(cls) -> "BulletinConfig":
        load_dotenv()
        adapter = os.getenv("LORA_ADAPTER_PATH", "").strip()
        return cls(
            technical_excel_path=Path(
                os.getenv("TECHNICAL_EXCEL_PATH", "data/technical/checklist_tecnico.xlsx")
            ),
            technical_json_path=Path(
                os.getenv("TECHNICAL_JSON_PATH", "data/processed/technical/checklist_tecnico.json")
            ),
            output_dir=Path(
                os.getenv("BULLETIN_OUTPUT_DIR", "data/generated/bulletins")
            ),
            text_backend=os.getenv("BULLETIN_TEXT_BACKEND", "openai").strip().lower(),
            openai_model=os.getenv("BULLETIN_OPENAI_MODEL", "").strip(),
            openai_timeout_seconds=float(os.getenv("BULLETIN_OPENAI_TIMEOUT_SECONDS", "60")),
            llm_fallback_enabled=_env_bool("BULLETIN_LLM_FALLBACK", True),
            lora_base_model=os.getenv("LORA_BASE_MODEL", "").strip(),
            lora_adapter_path=Path(adapter) if adapter else None,
            history_days=int(os.getenv("BULLETIN_HISTORY_DAYS", "60")),
        )

    def validate(self) -> None:
        if self.history_days <= 1:
            raise ValueError("BULLETIN_HISTORY_DAYS debe ser mayor que uno")
        if self.openai_timeout_seconds <= 0:
            raise ValueError("BULLETIN_OPENAI_TIMEOUT_SECONDS debe ser positivo")
        if self.text_backend not in {"template", "openai", "lora"}:
            raise ValueError("BULLETIN_TEXT_BACKEND debe ser template, openai o lora")
        if self.text_backend == "lora":
            if not self.lora_base_model:
                raise ValueError("Falta LORA_BASE_MODEL")
            if self.lora_adapter_path is None:
                raise ValueError("Falta LORA_ADAPTER_PATH")


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().casefold() in {"1", "true", "yes", "si", "sí", "on"}
