"""Prepara ejemplos validados para SFT sin mezclar fechas entre splits."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


SYSTEM = (
    "Redacta boletines tecnicos de mercado usando solo el contexto suministrado. "
    "La salida debe ser JSON con title, classification, executive_summary, "
    "highlights y sections. No inventes precios ni noticias."
)


def load_examples(path: Path, *, allow_synthetic: bool = False) -> list[dict[str, object]]:
    examples: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        _validate_example(value, line_number)
        if value.get("validated") is True or (
            allow_synthetic and value.get("synthetic") is True
        ):
            examples.append(value)
    if not examples:
        raise ValueError(
            "No hay boletines historicos validados. Usa --allow-synthetic solo para pruebas."
        )
    return examples


def prepare(
    source: Path,
    output_dir: Path,
    *,
    validation_ratio: float = 0.2,
    seed: int = 42,
    allow_synthetic: bool = False,
) -> tuple[Path, Path]:
    if not 0 < validation_ratio < 1:
        raise ValueError("validation_ratio debe estar entre cero y uno")
    examples = load_examples(source, allow_synthetic=allow_synthetic)
    dates = sorted({str(item["date"]) for item in examples})
    if len(dates) < 2:
        raise ValueError("Se necesitan al menos dos fechas para separar train y validacion")
    random.Random(seed).shuffle(dates)
    validation_count = max(1, round(len(dates) * validation_ratio))
    validation_dates = set(dates[:validation_count])
    train = [_format(item) for item in examples if item["date"] not in validation_dates]
    validation = [_format(item) for item in examples if item["date"] in validation_dates]
    if not train or not validation:
        raise ValueError("Los splits de entrenamiento y validacion no pueden estar vacios")

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.jsonl"
    validation_path = output_dir / "validation.jsonl"
    _write_jsonl(train_path, train)
    _write_jsonl(validation_path, validation)
    return train_path, validation_path


def _validate_example(value: dict[str, object], line_number: int) -> None:
    required = {"id", "date", "input", "output", "validated", "synthetic"}
    missing = required.difference(value)
    if missing:
        raise ValueError(f"Linea {line_number}: faltan campos {sorted(missing)}")
    output = value["output"]
    if not isinstance(output, dict):
        raise ValueError(f"Linea {line_number}: output debe ser un objeto")
    required_output = {
        "title",
        "classification",
        "executive_summary",
        "highlights",
        "sections",
    }
    missing_output = required_output.difference(output)
    if missing_output:
        raise ValueError(
            f"Linea {line_number}: faltan campos de salida {sorted(missing_output)}"
        )
    if output["classification"] not in {"Alcista", "Bajista", "Neutral"}:
        raise ValueError(f"Linea {line_number}: classification invalida")


def _format(value: dict[str, object]) -> dict[str, object]:
    user = json.dumps(value["input"], ensure_ascii=False, sort_keys=True)
    assistant = json.dumps(value["output"], ensure_ascii=False, sort_keys=True)
    text = (
        f"<|system|>\n{SYSTEM}\n"
        f"<|user|>\n{user}\n"
        f"<|assistant|>\n{assistant}"
    )
    return {
        "id": value["id"],
        "date": value["date"],
        "text": text,
        "synthetic": value["synthetic"],
    }


def _write_jsonl(path: Path, values: list[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in values) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepara dataset JSONL para LoRA")
    parser.add_argument("--source", type=Path, default=Path("training/data/examples.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("training/data/prepared"))
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-synthetic", action="store_true")
    args = parser.parse_args()
    train, validation = prepare(
        args.source,
        args.output_dir,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
        allow_synthetic=args.allow_synthetic,
    )
    print(json.dumps({"train": str(train), "validation": str(validation)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
