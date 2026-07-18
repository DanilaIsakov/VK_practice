# Результаты оценки

Заполните после запуска `python scripts/evaluate_lmms.py`.

## Протокол

- Фреймворк: `lmms-eval`
- Задачи: `gqa-ru`, `mmbench_ru_dev`
- Batch size: 1
- OpenAI API для MMBench: нет (ExactMatch) / да (GPTEvalScore) — указать

## Baseline (публично, deepvk)

| Модель | GQA-ru | MMBench-ru |
| --- | --- | --- |
| Intel/llava-gemma-2b | 0.20 | 28.30 |
| deepvk/llava-gemma-2b-lora | 46.37 | 40.19 |
| deepvk/llava-saiga-8b | 51.44 | 56.65 |

## Наша модель

| Checkpoint | GQA-ru ExactMatch | MMBench-ru | Дата | Примечание |
| --- | --- | --- | --- | --- |
| `outputs/ruvlm-gemma-2b-lora` | | | | |

## Наблюдения

- Что улучшилось:
- Что ухудшилось:
- Гипотезы на следующий прогон:
