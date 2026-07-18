#!/usr/bin/env python3
"""Подготовка обучающих примеров из датасетов deepvk."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llava-dataset", default="deepvk/LLaVA-Instruct-ru")
    parser.add_argument("--gqa-dataset", default="deepvk/GQA-ru")
    parser.add_argument("--gqa-post-prompt", default=" Ответь одним словом.")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="data/train_mix.jsonl")
    return parser.parse_args()


def conversations_to_messages(conversations: list[dict[str, str]]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for turn in conversations:
        role = "user" if turn.get("from") in {"human", "user"} else "assistant"
        messages.append({"role": role, "content": turn["value"]})
    return messages


def from_llava_instruct(dataset_name: str, max_samples: int | None, seed: int) -> list[dict[str, Any]]:
    ds = load_dataset(dataset_name, split="train")
    if max_samples is not None:
        ds = ds.shuffle(seed=seed).select(range(min(max_samples, len(ds))))

    rows: list[dict[str, Any]] = []
    for item in ds:
        rows.append(
            {
                "source": "llava_instruct_ru",
                "id": item.get("id"),
                "image": item.get("image"),
                "messages": conversations_to_messages(item["conversations"]),
            }
        )
    return rows


def from_gqa_train(
    dataset_name: str,
    post_prompt: str,
    max_samples: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    instr = load_dataset(dataset_name, "train_balanced_instructions", split="train")
    if max_samples is not None:
        instr = instr.shuffle(seed=seed).select(range(min(max_samples, len(instr))))

    needed_ids = set(instr["imageId"])
    imgs = load_dataset(dataset_name, "train_balanced_images", split="train")
    imgs = imgs.filter(lambda x: x["id"] in needed_ids)
    img_by_id = {row["id"]: row["image"] for row in imgs}

    rows: list[dict[str, Any]] = []
    for item in instr:
        question = item["question"].rstrip()
        if post_prompt and not question.endswith(post_prompt.strip()):
            question = f"{question}{post_prompt}"
        answer = item.get("answer") or item.get("fullAnswer") or ""
        rows.append(
            {
                "source": "gqa_ru",
                "id": item.get("id"),
                "image": img_by_id[item["imageId"]],
                "messages": [
                    {"role": "user", "content": f"<image>\n{question}"},
                    {"role": "assistant", "content": str(answer)},
                ],
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Если указан max_samples — делим лимит пополам между источниками
    per_source = None if args.max_samples is None else max(1, args.max_samples // 2)

    rows = []
    rows.extend(from_llava_instruct(args.llava_dataset, per_source, args.seed))
    rows.extend(from_gqa_train(args.gqa_dataset, args.gqa_post_prompt, per_source, args.seed))

    shuffled = Dataset.from_list(rows).shuffle(seed=args.seed)

    with out_path.open("w", encoding="utf-8") as f:
        for row in shuffled:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved {len(shuffled)} examples → {out_path}")


if __name__ == "__main__":
    main()
