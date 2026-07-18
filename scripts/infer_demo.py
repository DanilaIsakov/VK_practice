#!/usr/bin/env python3
"""Демонстрационный инференс русскоязычной VLM."""

from __future__ import annotations

import argparse

import requests
import torch
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer, LlavaForConditionalGeneration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="deepvk/llava-gemma-2b-lora")
    parser.add_argument(
        "--image",
        default="https://www.ilankelman.org/stopsigns/australia.jpg",
        help="URL или локальный путь к изображению",
    )
    parser.add_argument(
        "--prompt",
        default="Опиши картинку несколькими словами.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    return parser.parse_args()


def load_image(path_or_url: str) -> Image.Image:
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        return Image.open(requests.get(path_or_url, stream=True).raw).convert("RGB")
    return Image.open(path_or_url).convert("RGB")


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LlavaForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
    ).to(device)
    processor = AutoProcessor.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    image = load_image(args.image)
    messages = [{"role": "user", "content": f"<image>\n{args.prompt}"}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(images=[image], text=text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=args.max_new_tokens)

    answer = tokenizer.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    print(answer.strip())


if __name__ == "__main__":
    main()
