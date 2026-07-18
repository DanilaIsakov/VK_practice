#!/usr/bin/env python3
"""LoRA instruction tuning русскоязычной VLM на данных deepvk."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
    LlavaForConditionalGeneration,
    TrainingArguments,
)
from trl import SFTTrainer


def load_config(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train_lora.yaml")
    parser.add_argument(
        "--data-file",
        default=None,
        help="JSONL из prepare_data.py. Если не задан — данные грузятся онлайн в упрощённом режиме.",
    )
    return parser.parse_args()


LLAVA_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}USER: {% else %}ASSISTANT: {% endif %}"
    "{{ message['content'] }}"
    "{% if message['role'] != 'user' %}{{ eos_token }}{% else %} {% endif %}"
    "{% endfor %}"
)


class VLMDataCollator:
    def __init__(self, processor: AutoProcessor):
        self.processor = processor

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        texts: list[str] = []
        images: list[Any] = []

        for example in examples:
            messages = example["messages"]
            # Упрощаем content к строке, если вдруг пришёл список частей
            normalized = []
            for m in messages:
                content = m["content"]
                if isinstance(content, list):
                    content = " ".join(
                        part.get("text", "") if isinstance(part, dict) else str(part) for part in content
                    )
                normalized.append({"role": m["role"], "content": content})

            text = self.processor.tokenizer.apply_chat_template(
                normalized,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)

            image = example.get("image")
            images.append(image)

        # Если изображений нет в локальном JSONL (только пути) — обучаем text-only fallback
        if all(img is None or isinstance(img, (str, int)) for img in images):
            batch = self.processor.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
        else:
            batch = self.processor(text=texts, images=images, return_tensors="pt", padding=True)

        labels = batch["input_ids"].clone()
        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100
        batch["labels"] = labels
        return batch


def build_online_dataset(cfg: dict[str, Any]):
    """Упрощённый онлайн-микс без локального JSONL (GQA-ru train)."""
    gqa = load_dataset(cfg["gqa_dataset"], split="train")
    max_samples = cfg.get("max_samples")
    if max_samples:
        gqa = gqa.shuffle(seed=cfg.get("seed", 42)).select(range(min(max_samples, len(gqa))))

    post = cfg.get("gqa_post_prompt", " Ответь одним словом.")

    def map_row(example: dict[str, Any]) -> dict[str, Any]:
        question = example["question"].rstrip() + post
        answer = example.get("answer") or example.get("fullAnswer") or ""
        return {
            "messages": [
                {"role": "user", "content": f"<image>\n{question}"},
                {"role": "assistant", "content": str(answer)},
            ],
            "image": example.get("image"),
        }

    return gqa.map(map_row, remove_columns=gqa.column_names)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    quant_config = None
    if cfg.get("load_in_4bit"):
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if cfg.get("bf16") else torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    model = LlavaForConditionalGeneration.from_pretrained(
        cfg["model_name_or_path"],
        torch_dtype=torch.bfloat16 if cfg.get("bf16") else torch.float16,
        quantization_config=quant_config,
        low_cpu_mem_usage=True,
    )
    processor = AutoProcessor.from_pretrained(cfg["model_name_or_path"])
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name_or_path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not tokenizer.chat_template:
        tokenizer.chat_template = LLAVA_CHAT_TEMPLATE
    processor.tokenizer = tokenizer

    if cfg.get("load_in_4bit"):
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["lora_target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    if args.data_file:
        train_dataset = load_dataset("json", data_files=args.data_file, split="train")
    else:
        train_dataset = build_online_dataset(cfg)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=cfg["num_train_epochs"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        warmup_ratio=cfg["warmup_ratio"],
        weight_decay=cfg["weight_decay"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        logging_steps=cfg["logging_steps"],
        save_steps=cfg["save_steps"],
        save_total_limit=cfg["save_total_limit"],
        bf16=cfg.get("bf16", False),
        fp16=cfg.get("fp16", False),
        gradient_checkpointing=cfg.get("gradient_checkpointing", True),
        dataloader_num_workers=cfg.get("dataloader_num_workers", 0),
        report_to=cfg.get("report_to", "none"),
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=VLMDataCollator(processor),
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"Checkpoint saved to {output_dir}")


if __name__ == "__main__":
    main()
