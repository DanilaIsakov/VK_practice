# Подробное описание решения

## 1. Постановка задачи

Необходимо дообучить Vision-Language Model (VLM) на открытых русскоязычных данных deepvk и добиться высоких метрик на бенчмарках **GQA-ru** и **MMBench-ru**.

Формулировка результата: модель принимает **изображение + текстовый вопрос на русском** и генерирует **текстовый ответ**.

## 2. Выбор подхода

### Почему VLM в стиле LLaVA

Типичная VLM состоит из трёх блоков ([статья Hugging Face](https://huggingface.co/blog/vlms)):

1. **Image encoder** (CLIP / SigLIP) — извлекает визуальные признаки.
2. **Multimodal projector** — выравнивает визуальные эмбеддинги с пространством LLM.
3. **Text decoder (LLM)** — генерирует ответ.

Схема LLaVA оптимальна для учебного проекта:

- можно заморозить encoder и обучать projector + (частично) LLM;
- поддерживается **LoRA / QLoRA** — обучение на одной GPU;
- совместима с `transformers` и `lmms-eval`.

### Базовая модель

| Вариант | Плюсы | Минусы |
| --- | --- | --- |
| `deepvk/llava-gemma-2b-lora` | Уже адаптирована под русский, компактная (~3B) | Ниже потолок качества, чем у 7–8B |
| `deepvk/llava-saiga-8b` | Сильнее на бенчмарках | Требует больше VRAM |
| Английская LLaVA с нуля | Полный контроль | Дольше и дороже доводить до русского |

**Выбранный путь проекта:** взять `google/gemma-2b-it` / `deepvk/llava-gemma-2b-lora` и провести **instruction tuning с LoRA** на данных deepvk, затем измерить прирост на GQA-ru / MMBench-ru.

Ориентир по публичным результатам deepvk:

| Модель | GQA-ru | MMBench-ru |
| --- | --- | --- |
| Intel/llava-gemma-2b | 0.20 | 28.30 |
| deepvk/llava-gemma-2b-lora | 46.37 | 40.19 |
| deepvk/llava-saiga-8b | 51.44 | 56.65 |
| llava-hf/llava-1.5-7b-hf | 28.39 | 52.25 |

Цель компактной модели — превзойти или приблизиться к `llava-gemma-2b-lora`, зафиксировав воспроизводимый пайплайн.

## 3. Данные deepvk: как именно использовались

### 3.1. Финальный прогон — `deepvk/GQA-ru`

Основной (и единственный в финальном Colab-обучении) датасет:

| Config | Роль | Объём |
| --- | --- | --- |
| `train_balanced_instructions` | вопросы/ответы для SFT | 40k (взято ~2.5k) |
| `train_balanced_images` | изображения по `imageId` | 27.5k (только нужные id) |
| `testdev_*` | оценка ExactMatch | не в train |

Предобработка:

1. Раздельная загрузка instructions и images (обязательный `config_name`).
2. Subset + join по `imageId` **без** `.filter()` по всем 27k (иначе OOM в Colab).
3. Пост-промпт: ` Ответь одним словом.`
4. Chat-template LLaVA: `<image>\n{вопрос}` → краткий `answer`.

Источник и лицензия: Hugging Face `deepvk/GQA-ru` (перевод GPT-4-turbo + фильтрация deepvk).

### 3.2. Другие датасеты коллекции (в финальном train не использовались)

| Датасет | Зачем в проекте |
| --- | --- |
| `deepvk/LLaVA-Instruct-ru` | запланирован для полного instruction tuning; отложен из‑за RAM/времени |
| `deepvk/MMBench-ru` | **только оценка** (не train), чтобы не завышать метрику |

### 3.3. Подготовка батча

1. Изображение RGB + текст вопроса.
2. `apply_chat_template` + `AutoProcessor`.
3. `labels`: pad → `-100`.

Скрипты: `scripts/prepare_data.py`, `notebooks/train_colab.ipynb`.

## 4. Процесс обучения и гиперпараметры

### 4.1. Схема

```
GQA-ru subset → chat examples
        ↓
deepvk/llava-gemma-2b-lora (база)
        ↓
LoRA SFT (q_proj, v_proj), fp16, Colab T4
        ↓
adapter ~26 MB + processor/tokenizer
```

Vision encoder не дообучался отдельно; адаптировался LLM через LoRA.

### 4.2. Фактические гиперпараметры финала

| Параметр | Значение |
| --- | --- |
| LoRA r / alpha / dropout | 32 / 32 / 0.05 |
| Target modules | `q_proj`, `v_proj` |
| LR | cosine, peak 1e-4 |
| Epochs / steps | 2 / 624 |
| Batch / grad accum | 1 / 8 |
| Precision | fp16 |
| Gradient checkpointing | да |
| PEFT | 0.13.2 |
| Train loss | 12.17 → 1.26 |

Конфиг-шаблон: `configs/train_lora.yaml`.  
Обучение: `scripts/train_lora.py` / Colab-ноутбук.

### 4.3. Оценка

Целевой протокол — `lmms-eval` (`gqa-ru`, `mmbench_ru_dev`), обёртка `scripts/evaluate_lmms.py`.

В сдаче зафиксированы train-метрики, качественное демо и сравнение с **опубликованными** baseline deepvk. Полный ExactMatch на testdev — следующий шаг на GPU (см. [`results/metrics.md`](../results/metrics.md)).

## 5. Инференс (демонстрация)

Скрипт `scripts/infer_demo.py` и ячейки Colab:

- загружают базу + LoRA-адаптер;
- принимают изображение и вопрос на русском;
- печатают ответ.

Подтверждено качественное RU-описание внешнего фото (стоп-знак).

## 6. Ограничения и риски

- Subset GQA-ru и лимит Colab RAM ограничивают полноту тюнинга.
- Полный ExactMatch через `lmms-eval` требует отдельного GPU-прогона после обучения.
- **Метрика ExactMatch** строгая: синонимы считаются ошибкой.
- Соблюдайте лицензии Gemma и датасетов deepvk.

## 7. План работ (выполнен)

1. Изучены VLM / LLaVA и коллекция deepvk.
2. Собран пайплайн (скрипты + Colab).
3. Обучен LoRA-адаптер на GQA-ru.
4. Зафиксированы метрики обучения, сравнение с baseline, анализ и выводы в `results/metrics.md`.
5. Репозиторий: https://github.com/DanilaIsakov/VK_practice

## 8. Итог

Получен воспроизводимый результат:

**deepvk/GQA-ru → LoRA (r=32) на deepvk/llava-gemma-2b-lora → адаптер ~26 MB, loss 12.17→1.26.**

Подробный отчёт (датасет, гиперпараметры, сравнение, анализ, выводы):  
[`results/metrics.md`](../results/metrics.md).
