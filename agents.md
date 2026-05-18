# AudioSepAblationStudy — инструкции для Cline

Этот файл является аналогом `CLAUDE.md`, но предназначен для работы с Cline и другими AI-агентами, которые читают инструкции проекта из `agents.md`.

## Назначение проекта

Проект предназначен для ablation study моделей разделения аудио на базе Conv-TasNet:

- `baseline` — базовая Conv-TasNet;
- `dsc5`, `dsc10`, `dsc20`, `dsc40` — варианты модели с DSC-модификациями;
- результаты экспериментов сохраняются в `results/` в формате CSV.

## Структура проекта

```text
AudioSepAblationStudy/
├── agents.md
├── configs/
│   ├── baseline.yaml
│   ├── dsc5.yaml
│   ├── dsc10.yaml
│   ├── dsc20.yaml
│   └── dsc40.yaml
├── src/
│   ├── models/
│   │   ├── conv_tasnet.py
│   │   └── dsc_conv_tasnet.py
│   ├── data/
│   │   └── musdb_dataset.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
├── notebooks/
│   └── kaggle_experiment.ipynb
├── results/
│   └── .gitkeep
└── requirements.txt
```

## Правила работы для Cline

1. Не создавай `CLAUDE.md`; инструкции проекта должны храниться в `agents.md`.
2. Перед изменением архитектуры сначала изучай существующие файлы в `src/`, `configs/` и `notebooks/`.
3. Сохраняй совместимость конфигов: новые параметры должны иметь разумные значения по умолчанию.
4. Не коммить датасеты, checkpoint-файлы и большие артефакты обучения.
5. Все результаты экспериментов сохраняй в `results/`, желательно отдельным CSV для каждого запуска.
6. Код должен быть воспроизводимым: фиксируй `seed`, логируй конфиг запуска и метрики.

## Ожидаемый workflow

### Установка зависимостей

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

### Запуск обучения

```bash
python src/train.py --config configs/baseline.yaml
python src/train.py --config configs/dsc5.yaml
python src/train.py --config configs/dsc10.yaml
python src/train.py --config configs/dsc20.yaml
python src/train.py --config configs/dsc40.yaml
```

### Запуск оценки

```bash
python src/evaluate.py --config configs/baseline.yaml --checkpoint path/to/checkpoint.pt
```

## Конвенции разработки

- Модели находятся в `src/models/`.
- Код загрузки и нарезки датасета находится в `src/data/`.
- Общие функции — в `src/utils.py`.
- Все гиперпараметры экспериментов должны задаваться через YAML в `configs/`.
- Notebook в `notebooks/kaggle_experiment.ipynb` используется как удобная точка запуска экспериментов на Kaggle, но основная логика должна оставаться в `src/`.

## Метрики

Для music source separation рекомендуется сохранять как минимум:

- `experiment`;
- `seed`;
- `checkpoint`;
- `si_sdr`;
- `sdr`;
- `sir`;
- `sar`;
- `params`;
- `inference_time_ms`.

## Ограничения

- Не помещай аудиоданные MUSDB18 внутрь репозитория.
- Не смешивай экспериментальную логику notebook с production-кодом в `src/`.
- Если добавляется новая модель, она должна иметь отдельный конфиг и не ломать существующие конфиги.