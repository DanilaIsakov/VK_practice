# Model Card: RuVLM-Gemma-2B-LoRA

## Обзор модели

| Поле | Значение |
| --- | --- |
| Имя | `RuVLM-Gemma-2B-LoRA` (рабочее название проекта) |
| Тип | Vision-Language Model (image-text-to-text) |
| База | `deepvk/llava-gemma-2b-lora` / `google/gemma-2b-it` + CLIP vision tower |
| Язык | Русский (основной), английский (частично) |
| Метод адаптации | LoRA (Low-Rank Adaptation) |
| Назначение | Visual Question Answering и instruction-following по изображениям |

## Для чего обучена модель

Модель обучается, чтобы:

1. Понимать содержимое изображения.
2. Отвечать на **русскоязычные** вопросы о сцене, объектах, отношениях и атрибутах.
3. Следовать инструкциям (краткий ответ / развёрнутое рассуждение).
4. Показать конкурентоспособные метрики на **GQA-ru** и **MMBench-ru**.

## Архитектура

```
[Изображение] → Vision Encoder (CLIP, frozen)
                      ↓
              Multimodal Projector (+ LoRA)
                      ↓
[Текст вопроса] → Tokenizer → Embedding
                      ↓
                 LLM Decoder (Gemma-2B + LoRA)
                      ↓
                 Текстовый ответ
```

- В промпте изображение обозначается специальным тегом `<image>`.
- Диалог оформляется через chat-template модели.

## Данные обучения (открытые данные VK / deepvk)

| Датасет | Split | Использование |
| --- | --- | --- |
| [deepvk/LLaVA-Instruct-ru](https://huggingface.co/datasets/deepvk/LLaVA-Instruct-ru) | train | Instruction tuning (диалоги, reasoning) |
| [deepvk/GQA-ru](https://huggingface.co/datasets/deepvk/GQA-ru) | train | Short-answer VQA (+ «Ответь одним словом.») |

Изображения для Instruct-части берутся из COCO (как в исходном пайплайне LLaVA).

## Данные оценки (не в обучении для MMBench)

| Датасет | Метрика |
| --- | --- |
| [deepvk/GQA-ru](https://huggingface.co/datasets/deepvk/GQA-ru) test | ExactMatch |
| [deepvk/MMBench-ru](https://huggingface.co/datasets/deepvk/MMBench-ru) | ExactMatch / GPTEvalScore |

## Гиперпараметры обучения

Заполните после фактического прогона (шаблон):

| Параметр | Значение |
| --- | --- |
| LoRA r / alpha / dropout | 32 / 64 / 0.05 |
| Target modules | `q_proj`, `v_proj`, `k_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Learning rate | 1e-4 |
| Warmup ratio | 0.03 |
| Weight decay | 0.0 |
| Epochs | 1 |
| Effective batch size | 16 |
| Precision | bf16 |
| Max seq length | 2048 |
| Hardware | (указать GPU) |
| Training time | (указать) |

## Результаты

Публичные baseline (deepvk):

| Модель | GQA-ru | MMBench-ru |
| --- | --- | --- |
| Intel/llava-gemma-2b | 0.20 | 28.30 |
| deepvk/llava-gemma-2b-lora | 46.37 | 40.19 |
| deepvk/llava-saiga-8b | 51.44 | 56.65 |

Результаты **этой** модели после обучения:

| Метрика | Значение |
| --- | --- |
| GQA-ru ExactMatch | _заполнить после `evaluate_lmms.py`_ |
| MMBench-ru ExactMatch | _заполнить после `evaluate_lmms.py`_ |

Подробный лог: `results/metrics.md`, сырые логи lmms-eval: `logs/`.

## Пример использования

```python
import requests
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer, LlavaForConditionalGeneration

model_name = "PATH_TO_YOUR_CHECKPOINT"  # или deepvk/llava-gemma-2b-lora

model = LlavaForConditionalGeneration.from_pretrained(model_name)
processor = AutoProcessor.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

url = "https://www.ilankelman.org/stopsigns/australia.jpg"
img = Image.open(requests.get(url, stream=True).raw)
messages = [
    {"role": "user", "content": "<image>\nОпиши картинку несколькими словами."}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(images=[img], text=text, return_tensors="pt")
generate_ids = model.generate(**inputs, max_new_tokens=64)
answer = tokenizer.decode(generate_ids[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
print(answer)
```

Демо-скрипт проекта: `scripts/infer_demo.py`.

## Ограничения

- Модель может ошибаться на мелком тексте (OCR), редких объектах и сложных пространственных отношениях.
- ExactMatch на GQA-ru не учитывает синонимы.
- Компактный размер (~2–3B) ограничивает сложные рассуждения по сравнению с 7–8B моделями.
- Не предназначена для критических решений (медицина, безопасность) без дополнительной валидации.

## Этические замечания

Обучающие диалоги сгенерированы LLM и могут содержать галлюцинации. Перед продуктовым использованием нужна фильтрация токсичности и проверка на предвзятость.

## Цитирование

```
@misc{deepvk2024vlm,
  title={Vision-Language Modeling (deepvk)},
  author={Belopolskih, Daniil and Spirin, Egor},
  url={https://huggingface.co/collections/deepvk/vision-language-modeling-664dd7e4c257cc78e740f6bc},
  year={2024}
}
```

```
@misc{liu2023llava,
  title={Visual Instruction Tuning},
  author={Liu, Haotian and Li, Chunyuan and Wu, Qingyang and Lee, Yong Jae},
  publisher={NeurIPS},
  year={2023}
}
```
