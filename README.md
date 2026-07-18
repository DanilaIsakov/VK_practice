# RuVLM BenchMax — русскоязычная VLM на данных deepvk

Учебный проект: дообучение Vision-Language модели на открытых датасетах VK ([deepvk](https://huggingface.co/collections/deepvk/vision-language-modeling-664dd7e4c257cc78e740f6bc)) и максимизация метрик на **GQA-ru** и **MMBench-ru**.

## Цель

Получить наиболее высокие метрики русскоязычной VLM на бенчмарках GQA-ru (ExactMatch) и MMBench-ru (ExactMatch / GPTEvalScore).

## Быстрый старт

### Локально (нужна NVIDIA GPU + CUDA PyTorch)

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

pip install -r requirements.txt

# Демо baseline-модели deepvk
python scripts/infer_demo.py --prompt "Что изображено на фото?"

# Подготовка смеси данных (опционально, для отладки max_samples)
python scripts/prepare_data.py --max-samples 2000 --output data/train_mix.jsonl

# LoRA-дообучение
python scripts/train_lora.py --config configs/train_lora.yaml

# Оценка (нужны GPU + lmms-eval)
python scripts/evaluate_lmms.py --model outputs/ruvlm-gemma-2b-lora
```

### Google Colab (рекомендуется без локальной GPU)

Откройте [`notebooks/train_colab.ipynb`](notebooks/train_colab.ipynb) в Colab, выберите Runtime → **T4 GPU** и запустите ячейки.

## Как используются данные VK

| Датасет | Роль |
| --- | --- |
| `deepvk/LLaVA-Instruct-ru` | Instruction tuning |
| `deepvk/GQA-ru` (train) | Обучение short-answer VQA |
| `deepvk/GQA-ru` (test) | Метрика ExactMatch |
| `deepvk/MMBench-ru` | Метрика множественного выбора |

Подробности: [docs/PROJECT_DESCRIPTION.md](docs/PROJECT_DESCRIPTION.md), [docs/SOLUTION.md](docs/SOLUTION.md).

## Материалы для сдачи

| Требование | Файл |
| --- | --- |
| Цель, задачи, ожидаемые результаты | [docs/PROJECT_DESCRIPTION.md](docs/PROJECT_DESCRIPTION.md) |
| Подробное описание решения | [docs/SOLUTION.md](docs/SOLUTION.md) |
| Описание обученной модели | [docs/MODEL_CARD.md](docs/MODEL_CARD.md) |
| Презентация (опционально) | [docs/PRESENTATION.md](docs/PRESENTATION.md) |
| Таблица метрик | [results/metrics.md](results/metrics.md) |

## Baseline (публичные результаты deepvk)

| Модель | GQA-ru | MMBench-ru |
| --- | --- | --- |
| Intel/llava-gemma-2b | 0.20 | 28.30 |
| deepvk/llava-gemma-2b-lora | 46.37 | 40.19 |
| deepvk/llava-saiga-8b | 51.44 | 56.65 |

## Структура репозитория

```
configs/train_lora.yaml      # гиперпараметры
scripts/prepare_data.py      # сбор train-микса
scripts/train_lora.py        # LoRA fine-tune
scripts/evaluate_lmms.py     # оценка lmms-eval
scripts/infer_demo.py        # демо инференса
docs/                        # документы проекта
results/metrics.md           # ваши результаты
```

## Ссылки

- Коллекция датасетов и моделей: https://huggingface.co/collections/deepvk/vision-language-modeling-664dd7e4c257cc78e740f6bc
- Статья про VLM: https://huggingface.co/blog/vlms
- GQA-ru: https://huggingface.co/datasets/deepvk/GQA-ru
- MMBench-ru: https://huggingface.co/datasets/deepvk/MMBench-ru
