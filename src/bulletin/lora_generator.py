"""Carga opcional de un modelo open-weight con adaptador PEFT/LoRA."""

from __future__ import annotations

import json
from pathlib import Path

from .models import BulletinDraft, UnifiedBulletinContext


class LoRABulletinGenerator:
    def __init__(
        self,
        base_model: str,
        adapter_path: Path,
        *,
        max_new_tokens: int = 1200,
    ) -> None:
        if not adapter_path.is_dir():
            raise FileNotFoundError(f"No existe el adaptador LoRA: {adapter_path}")
        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Instala requirements-training.txt para utilizar el backend LoRA"
            ) from exc

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype="auto",
            device_map="auto",
        )
        self.model = PeftModel.from_pretrained(base, str(adapter_path))
        self.model.eval()
        self.max_new_tokens = max_new_tokens

    def generate(self, context: UnifiedBulletinContext) -> BulletinDraft:
        prompt = _prompt(context)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        generated = output[0][inputs["input_ids"].shape[1] :]
        payload = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        try:
            value = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValueError("El adaptador LoRA no devolvio JSON valido") from exc
        return BulletinDraft(
            title=value["title"],
            classification=value["classification"],
            executive_summary=value["executive_summary"],
            highlights=tuple(value.get("highlights", [])),
            sections=tuple(value.get("sections", [])),
            backend="lora",
        )


def _prompt(context: UnifiedBulletinContext) -> str:
    return (
        "Genera un boletin ejecutivo en JSON usando solo el contexto. "
        "No memorices ni inventes precios o noticias. Campos obligatorios: "
        "title, classification (Alcista/Bajista/Neutral), executive_summary, "
        "highlights y sections.\nCONTEXTO:\n"
        + json.dumps(context.to_dict(), ensure_ascii=False)
        + "\nJSON:"
    )
