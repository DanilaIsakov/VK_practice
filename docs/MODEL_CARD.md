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

| Датасет | Split / config | Использование в финальном прогоне |
| --- | --- | --- |
| [deepvk/GQA-ru](https://huggingface.co/datasets/deepvk/GQA-ru) | `train_balanced_instructions` + `train_balanced_images` | SFT (~2500 примеров/эпоху), пост-промпт «Ответь одним словом.» |
| [deepvk/MMBench-ru](https://huggingface.co/datasets/deepvk/MMBench-ru) | dev | только целевая оценка (не train) |
| [deepvk/LLaVA-Instruct-ru](https://huggingface.co/datasets/deepvk/LLaVA-Instruct-ru) | train | в финальном прогоне не использовался (лимит RAM) |

GQA-ru: перевод оригинального GQA (GPT-4-turbo + фильтрация deepvk); вопросы и картинки в раздельных config, связь по `imageId`.

## Данные оценки

| Датасет | Метрика | Статус |
| --- | --- | --- |
| GQA-ru testdev | ExactMatch | протокол `lmms-eval`; полный прогон — следующий шаг |
| MMBench-ru | ExactMatch | то же |

## Результаты и сравнение

| Модель | GQA-ru | MMBench-ru | Примечание |
| --- | ---: | ---: | --- |
| Intel/llava-gemma-2b | 0.20 | 28.30 | почти без RU |
| deepvk/llava-gemma-2b-lora (база) | 46.37 | 40.19 | публичный baseline |
| deepvk/llava-saiga-8b | 51.44 | 56.65 | более крупная модель |
| **RuVLM (наш адаптер)** | train **12.17→1.26**; демо OK | — | LoRA r=32, 2 эпохи на GQA-ru |

Анализ и выводы: [`results/metrics.md`](../results/metrics.md).

## Гиперпараметры обучения

Фактический финальный прогон (Colab T4, артефакт `ruvlm-outputs`, 2026-07-19):

| Параметр | Значение |
| --- | --- |
| LoRA r / alpha / dropout | **32 / 32 / 0.05** |
| Target modules | `q_proj`, `v_proj` |
| Learning rate | cosine schedule, peak ~1e-4 (из логов) |
| Epochs | **2** (`num_train_epochs=2`) |
| Global steps | **624** |
| Per-device batch | 1 (+ gradient accumulation, эфф. ~8) |
| Оценка размера данных | ~2500 примеров GQA-ru / эпоху |
| Precision | fp16 |
| Gradient checkpointing | да |
| PEFT | 0.13.2 |
| Hardware | Google Colab T4 |
| Размер адаптера | ~26 MB (`adapter_model.safetensors`) |

Ранний черновик: `checkpoint-32` (LoRA **r=16**, 32 step) — не финал.

## Результаты

Публичные baseline (deepvk):

| Модель | GQA-ru | MMBench-ru |
| --- | --- | --- |
| Intel/llava-gemma-2b | 0.20 | 28.30 |
| deepvk/llava-gemma-2b-lora | 46.37 | 40.19 |
| deepvk/llava-saiga-8b | 51.44 | 56.65 |

Результаты **этой** модели:

| Метрика | Значение |
| --- | --- |
| Train loss (624 steps, 2 epochs) | **12.17 → 1.26** (min 1.16 @ step 550) |
| Qualitative demo (RU caption) | стоп-знак описан корректно |
| Сравнение с базой deepvk | база: GQA-ru 46.37 / MMBench-ru 40.19 (публично); наш вклад — continued LoRA-tune |
| Артефакт | `ruvlm-outputs/outputs/ruvlm-gemma-2b-lora/` |

Подробный разбор: [`results/metrics.md`](../results/metrics.md).

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
