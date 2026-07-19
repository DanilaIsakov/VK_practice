# Описание проекта: русскоязычная Vision-Language модель

## Название

**RuVLM BenchMax** — дообучение и оценка русскоязычной визуально-языковой модели на открытых датасетах deepvk с целью максимизации метрик на бенчмарках **GQA-ru** и **MMBench-ru**.

## Актуальность

Большинство открытых Vision-Language Models (VLM) обучаются преимущественно на английских данных. На русских вопросах к изображениям их качество резко падает: например, `Intel/llava-gemma-2b` показывает ~0.2 на GQA-ru при ~59.8 на английском GQA. Компания VK (команда deepvk) опубликовала русские датасеты и модели, что позволяет воспроизводимо исследовать и улучшать VLM именно для русского языка.

## Цель проекта

Получить **максимально высокие метрики** русскоязычной VLM на бенчмарках:

| Бенчмарк | Основная метрика | Что измеряет |
| --- | --- | --- |
| [GQA-ru](https://huggingface.co/datasets/deepvk/GQA-ru) | ExactMatch | Точность однословных ответов на вопросы о сцене |
| [MMBench-ru](https://huggingface.co/datasets/deepvk/MMBench-ru) | ExactMatch / GPTEvalScore | Многоаспектное понимание изображения (выбор из A/B/C/D) |

Практическая цель обучения: научить модель **отвечать на вопросы по изображению на русском языке** (описание, рассуждение, выбор варианта).

## Задачи

1. **Изучить** архитектуру VLM (encoder → projector → LLM) и пайплайн LLaVA по материалам [Vision Language Models Explained](https://huggingface.co/blog/vlms).
2. **Исследовать** открытую коллекцию deepvk: [Vision-Language Modeling](https://huggingface.co/collections/deepvk/vision-language-modeling-664dd7e4c257cc78e740f6bc).
3. **Выбрать** базовую модель и стратегию дообучения (LoRA / QLoRA), доступную на учебной GPU.
4. **Подготовить данные**:
   - обучение: `deepvk/LLaVA-Instruct-ru`, train-часть `deepvk/GQA-ru`;
   - оценка: test/dev `GQA-ru`, `MMBench-ru`.
5. **Дообучить** модель в стиле LLaVA (instruction tuning).
6. **Оценить** модель через `lmms-eval` на `gqa-ru` и `mmbench_ru_dev`.
7. **Сравнить** с публичными baseline deepvk и зафиксировать результаты.
8. **Оформить** материалы: описание проекта, описание решения, model card, скрипты.

## Как используются открытые данные VK (deepvk)

| Ресурс | Роль в проекте |
| --- | --- |
| `deepvk/LLaVA-Instruct-ru` | Основной корпус instruction-tuning (диалоги и complex reasoning на русском) |
| `deepvk/GQA-ru` (train) | Дообучение на visual QA; пост-промпт «Ответь одним словом.» |
| `deepvk/GQA-ru` (test) | Финальная метрика ExactMatch |
| `deepvk/MMBench-ru` | Финальная метрика по множественному выбору |
| `deepvk/llava-gemma-2b-lora` | Baseline / стартовая точка для сравнения и/или дальнейшего fine-tune |
| `deepvk/llava-saiga-8b` | Сильный reference baseline (~8B) |

Данные **не изменяются вручную**: загружаются с Hugging Face Hub, приводятся к формату чата LLaVA и подаются в trainer / `lmms-eval`.

## Ожидаемые результаты

1. **Рабочий пайплайн**: скрипты подготовки данных, LoRA-дообучения и оценки.
2. **Обученная модель** (LoRA-адаптер или полный чекпоинт) с описанием архитектуры и гиперпараметров.
3. **Таблица метрик** на GQA-ru и MMBench-ru относительно baseline.
4. **Целевые ориентиры** (для компактной модели ~2–3B с LoRA):
   - GQA-ru ExactMatch **≥ 46** (уровень `llava-gemma-2b-lora`) и стремление к **≥ 50**;
   - MMBench-ru **≥ 40** и стремление к **≥ 50**.
5. **Документы для сдачи**: описание проекта, подробное описание решения, model card.

## Образ результата (deliverables)

```
VK_practice/
├── README.md
├── docs/
│   ├── PROJECT_DESCRIPTION.md   ← цель, задачи, ожидаемые результаты
│   ├── SOLUTION.md              ← подробное описание решения
│   └── MODEL_CARD.md            ← описание обученной модели
├── configs/
│   └── train_lora.yaml
├── scripts/
│   ├── prepare_data.py
│   ├── train_lora.py
│   ├── evaluate_lmms.py
│   └── infer_demo.py
├── results/
│   └── metrics.md
└── requirements.txt
```
