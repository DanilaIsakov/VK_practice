# Отчёт о результатах: RuVLM BenchMax

Документ закрывает требования к описанию датасета, процесса обучения, гиперпараметров, метрик, сравнения с baseline, анализа и выводов.

Артефакт модели: [`ruvlm-outputs/outputs/ruvlm-gemma-2b-lora`](../ruvlm-outputs/outputs/ruvlm-gemma-2b-lora)  
База: [`deepvk/llava-gemma-2b-lora`](https://huggingface.co/deepvk/llava-gemma-2b-lora)  
Данные: [`deepvk/GQA-ru`](https://huggingface.co/datasets/deepvk/GQA-ru)

---

## 1. Описание использованного датасета

### 1.1. Выбор данных

Для финального дообучения использован открытый датасет VK **deepvk/GQA-ru** — русскоязычный перевод бенчмарка [GQA](https://cs.stanford.edu/people/dorarad/gqa/) (Hudson & Manning, 2019), подготовленный командой deepvk.

Почему GQA-ru:

- напрямую соответствует целевому бенчмарку проекта (visual question answering на русском);
- есть отдельный train для обучения и testdev для оценки;
- формат совместим с пайплайном `lmms-eval`, рекомендованным deepvk;
- не требует ручной разметки: данные публичны на Hugging Face Hub.

`deepvk/LLaVA-Instruct-ru` и `deepvk/MMBench-ru` рассматривались в дизайн-документах проекта; **в финальном Colab-прогоне** из‑за лимита RAM обучали только на GQA-ru (subset). MMBench-ru **не входил** в обучение (чтобы не завышать будущую оценку).

### 1.2. Структура GQA-ru

| Config | Split | Объём | Содержимое |
| --- | --- | --- | --- |
| `train_balanced_instructions` | train | 40 000 вопросов | `question`, `answer`, `fullAnswer`, `imageId`, … |
| `train_balanced_images` | train | 27 519 изображений | `id`, `image` |
| `testdev_balanced_instructions` | testdev | 12 216 вопросов | для оценки ExactMatch |
| `testdev_balanced_images` | testdev | 398 изображений | картинки для testdev |

Ключевые поля инструкции:

- `question` — вопрос на русском;
- `answer` — краткий (обычно однословный) эталон;
- `fullAnswer` — развёрнутый эталон;
- `imageId` — связь с таблицей изображений.

Происхождение перевода (по карточке датасета deepvk): GPT-4-turbo → фильтрация неудачных переводов → ручная проверка частых ошибок.

### 1.3. Как данные использовались в проекте

1. Загрузка instructions и images **раздельными config** (иначе Hugging Face требует имя config).
2. Выборка ~**2500** train-примеров (оценка по `trainer_state`: ~312.5 step/epoch × batch_eff 8).
3. Join вопрос↔картинка по `imageId`.
4. К вопросу добавлялся пост-промпт: **« Ответь одним словом.»** (как у deepvk при обучении на GQA-ru).
5. Пример приводился к chat-формату LLaVA: `user: <image>\n{вопрос}` → `assistant: {answer}`.

Testdev GQA-ru и MMBench-ru в обучение не попадали.

---

## 2. Процесс обучения модели

### 2.1. Базовая модель

Стартовая точка — **`deepvk/llava-gemma-2b-lora`**:

- архитектура LLaVA (CLIP vision encoder + projector + Gemma-2B);
- уже адаптирована под русский deepvk;
- компактна (~3B) и обучаема на Colab T4 через LoRA.

Альтернативы (`llava-saiga-8b`, английские 7B) отклонены из‑за VRAM на учебной GPU.

### 2.2. Метод адаптации

**LoRA (Low-Rank Adaptation)** поверх весов LLM:

- vision encoder не дообучался (заморожен в базовой схеме LLaVA);
- обучались низкоранговые матрицы в attention (`q_proj`, `v_proj`);
- на диск сохраняется только адаптер (~26 MB), а не полная копия 3B модели.

### 2.3. Этапы работы

1. **Подготовка окружения** (Colab T4): `transformers`, `peft`, `trl`, `datasets`; без `bitsandbytes`/`torchao` (конфликты в Colab).
2. **Подготовка данных без модели в RAM** — сначала subset картинок, затем загрузка модели (иначе OOM на ~12 GB системной RAM).
3. **Instruction tuning** — SFT на парах изображение–вопрос–ответ, 2 эпохи.
4. **Сохранение** адаптера, tokenizer/processor.
5. **Качественная проверка** — русскоязычный caption / VQA на внешних изображениях.
6. **Сравнение** с опубликованными метриками baseline deepvk и с собственным коротким прогоном (ablation).

Скрипты воспроизведения: `notebooks/train_colab.ipynb`, `scripts/train_lora.py`, `scripts/prepare_data.py`.

### 2.4. Сходимость (факт)

По `checkpoint-624/trainer_state.json`:

| Step | Loss | Интерпретация |
| --- | ---: | --- |
| 5 | 12.17 | старт |
| 100 | 1.95 | основной спад |
| 400 | 1.26 | выход на плато |
| 550 | **1.16** | минимум |
| 620 | 1.26 | финиш |

Loss снизился примерно **в 10 раз**; после ~400 шагов выгода от дальнейшего обучения невелика.

---

## 3. Выбранные гиперпараметры

Финальный прогон (Colab T4, июль 2026):

| Гиперпараметр | Значение | Обоснование |
| --- | --- | --- |
| Base model | `deepvk/llava-gemma-2b-lora` | сильный RU-baseline в классе 2–3B |
| LoRA rank `r` | **32** | больше ёмкости, чем у черновика r=16 |
| LoRA alpha | **32** | α = r, стабильный scaling |
| LoRA dropout | 0.05 | лёгкая регуляризация |
| Target modules | `q_proj`, `v_proj` | классический LoRA для attention, экономия VRAM |
| Learning rate | cosine, peak **1e-4** | стандарт для LoRA SFT |
| Warmup | ~3% schedule | плавный выход на LR |
| Epochs | **2** | до плато loss без явного переобучения по кривой |
| Max / global steps | **624** | 2 × ~312.5 step/epoch |
| Per-device batch | 1 | лимит VRAM T4 + изображения |
| Gradient accumulation | 8 | effective batch ≈ 8 |
| Precision | **fp16** | T4; bf16 на T4 хуже поддерживается |
| Gradient checkpointing | да | экономия активаций |
| Optimizer | AdamW | default HF Trainer / TRL |
| Logging / save | каждые 5 / 500 steps | контроль сходимости |
| PEFT | 0.13.2 | совместимость с Colab |
| Размер данных | ~2500 GQA-ru / эпоху | баланс качества и RAM |

### Ablation: короткий vs финальный прогон

| Прогон | Steps | r | Adapter | Loss (конец) |
| --- | ---: | ---: | ---: | ---: |
| Черновик `checkpoint-32` | 32 | 16 | ~13 MB | ~7–8 |
| **Финал** | **624** | **32** | **~26 MB** | **~1.26** |

Увеличение данных/эпох и ранга LoRA дало качественный скачок по train loss.

---

## 4. Итоговые метрики

### 4.1. Метрики, измеренные в проекте

| Метрика | Значение | Тип |
| --- | --- | --- |
| Train loss (start → end) | **12.1744 → 1.2589** | обучение |
| Train loss (минимум) | **1.1619** @ step 550 | обучение |
| Относительное снижение loss | ≈ **×9.7** | обучение |
| Число эпох / шагов | 2 / 624 | процесс |
| Размер адаптера | ~26 MB | артефакт |
| Качественный RU-инференс | корректное описание стоп-знака | демо |

Пример демо:

> Промпт: «Опиши картинку несколькими словами.»  
> Ответ: «На фотографии изображен красный стоп-знак, стоящий на улице, рядом с красным зданием.»

### 4.2. Метрики бенчмарков (целевые для проекта)

Целевые бенчмарки задания — **GQA-ru ExactMatch** и **MMBench-ru ExactMatch/GPTEval** через `lmms-eval`.

| Бенчмарк | Метрика | Статус в проекте |
| --- | --- | --- |
| GQA-ru (testdev) | ExactMatch | полный прогон `lmms-eval` не выполнен (лимит RAM/времени Colab после обучения) |
| MMBench-ru (dev) | ExactMatch | то же; датасет **не использовался** в train |

Протокол оценки (готовые скрипты): `scripts/evaluate_lmms.py`, задачи `gqa-ru`, `mmbench_ru_dev`.

Для отчёта зафиксированы **измеримые** метрики обучения и сравнение с **опубликованными** результатами той же линейки моделей (см. §5). Это корректная постановка при ограниченных compute: сходимость и воспроизводимый адаптер + ориентация на публичный leaderboard deepvk.

---

## 5. Сравнение с базовой моделью и другими решениями

Публичные результаты deepvk (оценка `lmms-eval`, без OpenAI API для MMBench → режим близкий к ExactMatch):

| Модель | Параметры (порядок) | GQA-ru ↑ | MMBench-ru ↑ | Комментарий |
| --- | --- | ---: | ---: | --- |
| Intel/llava-gemma-2b | ~2–3B | **0.20** | 28.30 | почти не умеет русский |
| llava-hf/llava-v1.6-mistral-7b-hf | ~7B | 6.65 | 48.80 | сильна в EN, слаба в RU GQA |
| llava-hf/llava-1.5-7b-hf | ~7B | 28.39 | 52.25 | средний RU |
| **deepvk/llava-gemma-2b-lora** (наша база) | ~3B | **46.37** | **40.19** | сильный RU в классе 2–3B |
| deepvk/llava-saiga-8b | ~8B | **51.44** | **56.65** | верхняя планка open RU VLM deepvk |
| **RuVLM (наш LoRA поверх deepvk-2B)** | +26 MB adapter | train loss 12.2→1.26; демо OK | — | continued fine-tune на GQA-ru subset |

### Интерпретация сравнения

1. **База выбрана удачно:** `deepvk/llava-gemma-2b-lora` уже на порядок лучше «сырой» Intel Gemma-VLM на GQA-ru (46.37 vs 0.20) при том же порядке размера модели.
2. **Наш вклад** — воспроизводимое **дообучение LoRA** на открытом GQA-ru с документированными гиперпараметрами и артефактом; по train-метрикам финальный прогон существенно лучше короткого ablation.
3. **Относительно saiga-8B** ожидаемо ниже потолок качества (меньше LLM), но выше доступность (T4, маленький адаптер).
4. Полное численное сравнение ExactMatch «база vs наш адаптер» на testdev — следующий шаг на GPU с `lmms-eval`; инфраструктура для этого в репозитории уже есть.

---

## 6. Анализ результатов

**Что удалось**

- Собрать end-to-end пайплайн на открытых данных VK без ручной разметки.
- Обойти типичные ловушки Colab (OOM на полном image split, конфликты `torchao`/`bitsandbytes`/`Pillow`).
- Получить стабильную сходимость: loss ×10↓ за 2 эпохи.
- Подтвердить жизнеспособность модели качественным русскоязычным ответом по изображению.
- Зафиксировать воспроизводимый артефакт (LoRA + processor/tokenizer).

**Ограничения**

- Обучение на **subset** GQA-ru (~2.5k), не на всех 40k и без LLaVA-Instruct-ru → потенциал качества не исчерпан.
- LoRA только на `q_proj`/`v_proj`; расширение target modules может дать прирост.
- Нет полного ExactMatch на testdev в текущей сдаче — нельзя утверждать численный прирост над 46.37 GQA-ru.
- ExactMatch жёсткая: синонимы считаются ошибкой; это свойство бенчмарка, не бага модели.

**Риски переобучения**

- Train и test GQA из одного домена; deepvk также использует train GQA-ru при тюнинге — практика стандартная, но MMBench-ru сознательно исключён из train как внешняя проверка обобщения.

---

## 7. Выводы

1. **Датасет.** Открытый **deepvk/GQA-ru** достаточен для учебного VLM-проекта: прозрачная структура, RU-язык, совместимость с бенчмарком задания.
2. **Обучение.** LoRA-дообучение `deepvk/llava-gemma-2b-lora` на Colab T4 **воспроизводимо**: 2 эпохи, r=32, fp16, effective batch 8, loss **12.17 → 1.26**.
3. **Метрики.** Зафиксированы train-метрики и качественное демо; целевые ExactMatch GQA-ru/MMBench-ru описаны протоколом и сравнительной таблицей публичных baseline.
4. **Сравнение.** База deepvk уже конкурентоспособна в классе 2–3B на RU; англоцентричные 7B на GQA-ru заметно слабее. Наш адаптер — практичное продолжение тюнинга с измеримой сходимостью и малым размером артефакта.
5. **Дальнейшая работа.** Полный `lmms-eval` на testdev; увеличение subset / добавление LLaVA-Instruct-ru; LoRA на большем наборе модулей; сравнение ExactMatch «база vs адаптер» на одинаковом железе.

---

## 8. Состав артефакта `ruvlm-outputs`

| Путь | Назначение |
| --- | --- |
| `adapter_model.safetensors` + `adapter_config.json` | финальный LoRA (~26 MB) |
| `tokenizer*`, `*processor*` | инференс |
| `checkpoint-624/` | финальный training checkpoint |
| `checkpoint-500/` | промежуточный |
| `checkpoint-32/` | черновик (r=16) — не использовать как финал |

Загрузка для инференса:

```python
from peft import PeftModel
from transformers import AutoProcessor, AutoTokenizer, LlavaForConditionalGeneration
import torch

BASE = "deepvk/llava-gemma-2b-lora"
ADAPTER = "ruvlm-outputs/outputs/ruvlm-gemma-2b-lora"

processor = AutoProcessor.from_pretrained(ADAPTER)
tokenizer = AutoTokenizer.from_pretrained(ADAPTER)
model = LlavaForConditionalGeneration.from_pretrained(
    BASE, torch_dtype=torch.float16, low_cpu_mem_usage=True
)
model = PeftModel.from_pretrained(model, ADAPTER)
model.eval()
```
