"""Orquestador RAG + Excel + generador editorial + PNG."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from rag.config import RAGConfig
from technical.excel_loader import load_technical_excel

from .config import BulletinConfig
from .context_builder import UnifiedContextBuilder
from .generator import BulletinTextGenerator, TemplateBulletinGenerator
from .guarded_generator import GuardedBulletinGenerator
from .lora_generator import LoRABulletinGenerator
from .models import BulletinResult
from .openai_generator import OpenAIBulletinGenerator
from .rag_adapter import CurrentRAGAdapter
from .renderer import BulletinRenderer


class BulletinPipeline:
    def __init__(self, bulletin_config: BulletinConfig, rag_config: RAGConfig, *,
                 generator: BulletinTextGenerator | None = None,
                 renderer: BulletinRenderer | None = None) -> None:
        bulletin_config.validate()
        self.config = bulletin_config
        self.rag_config = rag_config
        self.generator = generator or self._build_generator()
        self.renderer = renderer or BulletinRenderer()

    def _context_builder(self) -> UnifiedContextBuilder:
        technical = load_technical_excel(self.config.technical_excel_path)
        rag = CurrentRAGAdapter(self.rag_config.index_dir)
        return UnifiedContextBuilder(rag, technical, history_days=self.config.history_days)

    def available_dates(self) -> tuple[str, ...]:
        """Fechas disponibles simultáneamente en el RAG y en el Excel técnico."""
        return tuple(reversed(self._context_builder().available_dates))

    def run(self, publication_date: str | None = None, *, symbols: tuple[str, ...] | None = None) -> BulletinResult:
        technical = load_technical_excel(self.config.technical_excel_path)
        technical.save_json(self.config.technical_json_path)
        rag = CurrentRAGAdapter(self.rag_config.index_dir)
        builder = UnifiedContextBuilder(rag, technical, history_days=self.config.history_days)
        context = builder.build(publication_date, symbols=symbols)
        draft = self.generator.generate(context)
        output_dir = self.config.output_dir / context.publication_date
        output_dir.mkdir(parents=True, exist_ok=True)

        context_path = output_dir / "context.json"
        draft_path = output_dir / "draft.json"
        manifest_path = output_dir / "manifest.json"
        _write_json(context_path, context.to_dict())
        _write_json(draft_path, draft.to_dict())
        images = self.renderer.render(context, draft, output_dir)
        manifest = {
            "publication_date": context.publication_date,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "rag_index_dir": str(self.rag_config.index_dir),
            "technical_source": str(self.config.technical_excel_path),
            "technical_json": str(self.config.technical_json_path),
            "text_backend": draft.backend,
            "evidence_guardrails": True,
            "image_count": len(images),
            "images": [str(path) for path in images],
        }
        _write_json(manifest_path, manifest)
        return BulletinResult(publication_date=context.publication_date, images=images,
                              context_path=context_path, draft_path=draft_path,
                              manifest_path=manifest_path, backend=draft.backend)

    def _build_generator(self) -> BulletinTextGenerator:
        if self.config.text_backend == "template":
            return TemplateBulletinGenerator()
        if self.config.text_backend == "openai":
            fallback = TemplateBulletinGenerator() if self.config.llm_fallback_enabled else None
            primary = OpenAIBulletinGenerator(
                self.rag_config.openai_api_key,
                self.config.openai_model or self.rag_config.llm_model,
                timeout_seconds=self.config.openai_timeout_seconds,
                fallback=fallback,
            )
            return GuardedBulletinGenerator(primary)
        assert self.config.lora_adapter_path is not None
        return LoRABulletinGenerator(self.config.lora_base_model, self.config.lora_adapter_path)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
