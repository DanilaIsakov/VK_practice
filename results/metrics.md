# Результаты: анализ артефакта `ruvlm-outputs`

Источник: локальная папка [`ruvlm-outputs/outputs/ruvlm-gemma-2b-lora`](../ruvlm-outputs/outputs/ruvlm-gemma-2b-lora) (скачано из Colab).

## Состав артефакта

| Путь | Назначение | Размер |
| --- | --- | --- |
| `adapter_model.safetensors` + `adapter_config.json` | **Финальный LoRA-адаптер** (для инференса) | ~26 MB |
| `tokenizer*`, `preprocessor_config.json`, `processor_config.json` | Токенизатор / processor | ~37 MB |
| `checkpoint-624/` | Финальный training checkpoint (адаптер + optimizer) | ~26 MB adapter + ~28 MB optimizer |
| `checkpoint-500/` | Промежуточный (step 500) | аналогично |
| `checkpoint-32/` | Ранний прогон (**другой** LoRA: r=16, ~13 MB) | черновик |

Для сдачи и инференса достаточно **корня** `ruvlm-gemma-2b-lora/` (финальный адаптер).  
`checkpoint-32` — от короткого первого эксперимента, не путать с финалом.

## Конфигурация финального адаптера

Из `adapter_config.json`:

| Параметр | Значение |
| --- | --- |
| Base model | `deepvk/llava-gemma-2b-lora` |
| PEFT | LoRA (PEFT 0.13.2) |
| r / alpha / dropout | **32 / 32 / 0.05** |
| Target modules | `q_proj`, `v_proj` |
| Task | `CAUSAL_LM` |
| Inference mode | `true` (сохранённый адаптер) |

## Данные обучения

- Датасет: **deepvk/GQA-ru** (`train_balanced_instructions` + `train_balanced_images`)
- Пост-промпт: «Ответь одним словом.»
- Оценка размера выборки по `trainer_state.json`: ~**312.5 step / epoch** при `train_batch_size=1`  
  → при `gradient_accumulation_steps=8` ≈ **2500 примеров / эпоху**

## Ход обучения (из `checkpoint-624/trainer_state.json`)

| Параметр | Значение |
| --- | --- |
| `num_train_epochs` | **2** |
| `global_step` / `max_steps` | **624** |
| `epoch` (факт) | **1.9968** (~2 эпохи) |
| `logging_steps` | 5 |
| `save_steps` | 500 |
| Train loss (step 5 → 620) | **12.1744 → 1.2589** |
| Минимальный loss | **1.1619** (step 550) |
| `total_flos` | ~2.08e15 |

### Динамика loss

| Step | Epoch | Loss | Комментарий |
| --- | --- | --- | --- |
| 5 | 0.02 | 12.17 | старт |
| 50 | 0.16 | 4.87 | быстрый спуск |
| 100 | 0.32 | 1.95 | основной спад |
| 200 | 0.64 | 1.55 | замедление |
| 400 | 1.28 | 1.26 | плато |
| 550 | 1.76 | **1.16** | лучшая точка по loss |
| 620 | 1.98 | 1.26 | финиш (небольшой шум) |

**Вывод:** обучение стабильное, loss упал ~**×10**. После ~400 шагов прирост мал — 2 эпохи достаточно. Один лог с `grad_norm=NaN` (step 40) был единичным, на сходимость не повлиял.

## Сравнение прогонов

| Прогон | Steps | LoRA r | Adapter size | Loss end |
| --- | --- | --- | --- | --- |
| Короткий (`checkpoint-32`) | 32 | 16 | ~13 MB | ~7–8 (по логам Colab) |
| **Финал** (корень + `checkpoint-624`) | **624** | **32** | **~26 MB** | **~1.26** |

## Качественный инференс

Промпт: «Опиши картинку несколькими словами» (стоп-знак).

Ответ: *«На фотографии изображен красный стоп-знак, стоящий на улице, рядом с красным зданием.»*

Модель отвечает по-русски и по делу.

## Бенчмарки GQA-ru / MMBench-ru

| Метрика | Статус |
| --- | --- |
| GQA-ru ExactMatch | не замерено (`lmms-eval`) |
| MMBench-ru | не замерено |

Ориентиры deepvk (для сравнения после оценки):

| Модель | GQA-ru | MMBench-ru |
| --- | --- | --- |
| Intel/llava-gemma-2b | 0.20 | 28.30 |
| deepvk/llava-gemma-2b-lora | 46.37 | 40.19 |
| deepvk/llava-saiga-8b | 51.44 | 56.65 |

## Как загружать чекпоинт

```python
from peft import PeftModel
from transformers import AutoProcessor, AutoTokenizer, LlavaForConditionalGeneration
import torch

BASE = "deepvk/llava-gemma-2b-lora"
ADAPTER = "ruvlm-outputs/outputs/ruvlm-gemma-2b-lora"  # финальный корень

processor = AutoProcessor.from_pretrained(ADAPTER)
tokenizer = AutoTokenizer.from_pretrained(ADAPTER)
model = LlavaForConditionalGeneration.from_pretrained(
    BASE, torch_dtype=torch.float16, low_cpu_mem_usage=True
)
model = PeftModel.from_pretrained(model, ADAPTER)
model.eval()
```

## Итог для отчёта

1. Получен воспроизводимый LoRA-адаптер на открытых данных **deepvk/GQA-ru**.
2. Финальный прогон: **2 эпохи, 624 step, r=32**, loss **12.17 → 1.26**.
3. Артефакт готов к демонстрации; числовые бенчмарки — опциональный следующий шаг через `lmms-eval`.
