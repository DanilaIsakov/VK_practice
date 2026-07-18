#!/usr/bin/env python3
"""Запуск оценки на GQA-ru и MMBench-ru через lmms-eval."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="deepvk/llava-gemma-2b-lora",
        help="Путь к чекпоинту или имя модели на Hugging Face",
    )
    parser.add_argument(
        "--tasks",
        default="gqa-ru,mmbench_ru_dev",
        help="Список задач lmms-eval через запятую",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output-path", default="./logs")
    parser.add_argument("--suffix", default="ruvlm")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Только напечатать команду, не запускать",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    Path(args.output_path).mkdir(parents=True, exist_ok=True)

    cmd = [
        "accelerate",
        "launch",
        "-m",
        "lmms_eval",
        "--model",
        "llava_hf",
        "--model_args",
        f"pretrained={args.model}",
        "--tasks",
        args.tasks,
        "--batch_size",
        str(args.batch_size),
        "--log_samples",
        "--log_samples_suffix",
        args.suffix,
        "--output_path",
        args.output_path,
    ]

    print("Running:\n", " ".join(cmd))
    if args.dry_run:
        return

    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
