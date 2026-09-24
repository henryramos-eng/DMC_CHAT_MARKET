"""Entrenamiento PEFT/LoRA separado del runtime de Telegram."""

from __future__ import annotations

import argparse
from pathlib import Path


def train(
    base_model: str,
    train_file: Path,
    validation_file: Path,
    output_dir: Path,
    *,
    epochs: float = 3.0,
    learning_rate: float = 2e-4,
    max_length: int = 2048,
    target_modules: tuple[str, ...] = ("q_proj", "v_proj"),
) -> None:
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise RuntimeError("Instala requirements-training.txt antes de entrenar") from exc

    dataset = load_dataset(
        "json",
        data_files={
            "train": str(train_file),
            "validation": str(validation_file),
        },
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype="auto",
        device_map="auto",
    )
    model = get_peft_model(
        model,
        LoraConfig(
            task_type="CAUSAL_LM",
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=list(target_modules),
        ),
    )

    def tokenize(batch: dict[str, list[str]]) -> dict[str, object]:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
        )

    tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset["train"].column_names)
    arguments = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        load_best_model_at_end=True,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train()
    model.save_pretrained(output_dir / "adapter")
    tokenizer.save_pretrained(output_dir / "adapter")


def main() -> int:
    parser = argparse.ArgumentParser(description="Entrena el estilo del boletin con LoRA")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--train", type=Path, default=Path("training/data/prepared/train.jsonl"))
    parser.add_argument(
        "--validation",
        type=Path,
        default=Path("training/data/prepared/validation.jsonl"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("training/output"))
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--target-modules", default="q_proj,v_proj")
    args = parser.parse_args()
    train(
        args.base_model,
        args.train,
        args.validation,
        args.output_dir,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        target_modules=tuple(
            item.strip() for item in args.target_modules.split(",") if item.strip()
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
