# PADA_HRTECH_CANDIDATE_TP_PREDICTOR

**Прогнозирование риска увольнения сотрудников на основе карьерной истории резюме**

Гибридная нейросетевая система для оценки индивидуального риска досрочного увольнения кандидата. Использует исключительно данные резюме — без доступа к корпоративным HR-системам.

---

## Результаты

| Метод | Train C-index | Val C-index | 95% ДИ |
|---|---|---|---|
| **BiLSTM + Attention + DeepSurv** | **0.814** | **0.795** | [0.784; 0.806] |
| GBSA (Cox loss) | 0.669 | 0.648 | [0.633; 0.663] |
| Cox PH (L2) | 0.637 | 0.633 | [0.618; 0.647] |
| Cox PH + Elastic Net | 0.633 | 0.629 | [0.614; 0.644] |
| Weibull AFT | 0.631 | 0.629 | [0.614; 0.643] |
| Random Survival Forest | 0.716 | 0.619 | [0.605; 0.635] |
| Extra Survival Trees | 0.656 | 0.619 | [0.604; 0.633] |

**Превосходство над лучшим базовым методом (GBSA):** ΔC = +0.147, 95% ДИ [+0.134; +0.161], p < 0.001  
**Лог-ранговый критерий по группам риска:** χ² = 1630.2, p < 10⁻¹⁰⁰  
**Spearman(duration, risk):** −0.36 (train), −0.32 (val) — target leakage отсутствует

---

## Архитектура

```
CareerLSTMAttentionDeepSurv
├── BiLSTM encoder          (2 layers, hidden=128, bidirectional)
│     input: career sequence, T≤12, D=389
│     └── hash-vectorized text (384-dim) + 5 numeric job features
├── Additive attention      (softmax-weighted context vector, 256-dim)
├── Numeric branch          (22 static features, BatchNorm1d)
└── DeepSurv head           (278→128→64→1, Cox partial likelihood loss)

Trainable parameters: 1,037,230
```

**Ключевые детали:**
- Векторизация текста: хэш-векторизация Blake2b (без предобученных эмбеддингов)
- Обработка цензурированных наблюдений: Cox Partial Likelihood
- Защита от target leakage: длительность последнего эпизода маскируется **безусловно** для всех субъектов
- Функции выживания: оценка Бреслоу на горизонтах 6, 12, 24 месяца

---

## Данные

| Параметр | Значение |
|---|---|
| Исходный массив | 19 982 резюме |
| После фильтрации | 13 329 резюме |
| Train / Val | 10 663 / 2 666 (80/20, стратификация по event) |
| Event rate | 61.1% (завершённые эпизоды) |
| Медиана длительности | 32.0 мес. |
| seq_input_dim | 389 |
| numeric_dim | 22 |

Данные не публикуются (NDA). Схема маркировки: **self-supervised survival anchor** (последний хронологический эпизод → duration + event).

---

## Структура репозитория

```
├── notebooks/
│   └── vkr_supervised_lstm_attention_deepsurv_FIXED.ipynb   # основной ноутбук
├── src/
│   ├── features_real.py        # извлечение признаков и маркировка
│   ├── torch_model_v2.py       # архитектура BiLSTM+Attention+DeepSurv
│   └── metrics.py              # C-index, Breslow, survival_at_horizons
├── config.py                   # гиперпараметры
├── data/                       # пустая папка (данные не в репо)
├── models/                     # сохранённые веса (не в репо)
├── figures/                    # графики
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Быстрый старт

```bash
git clone https://github.com/yaale2/PADA_HRTECH_CANDIDATE_TP_PREDICTOR.git
cd PADA_HRTECH_CANDIDATE_TP_PREDICTOR
pip install -r requirements.txt
```

Откройте `notebooks/vkr_supervised_lstm_attention_deepsurv_FIXED.ipynb` в Google Colab.  
Поместите `careerist_resumes.jsonl` в `data/` (или Google Drive по пути из `config.py`).

---

## Гиперпараметры обучения

```python
HIDDEN_SIZE     = 128
NUM_LAYERS      = 2
BIDIRECTIONAL   = True
DROPOUT         = 0.3
BATCH_SIZE      = 512
EPOCHS          = 100          # ранняя остановка на эпохе 33
LR              = 3e-4
WARMUP_EPOCHS   = 5
EARLY_STOP_PATIENCE = 15
VAL_SHARE       = 0.2
SEED            = 42
```

---

## Топ признаков (permutation importance)

| Ранг | Признак | ΔC-index |
|---|---|---|
| 1 | has_higher_education | 0.0165 |
| 2 | history_has_any | 0.0050 |
| 3 | languages_count | 0.0031 |
| 4 | history_short_job_share_12m | 0.0029 |
| 5 | salary_per_history_experience_log | 0.0013 |

---

## Цитирование

```bibtex
@mastersthesis{ivanov2026,
  author    = {Иванов, Иван Иванович},
  title     = {Прогнозирование риска увольнения сотрудников на основе карьерной истории:
               гибридный подход с применением BiLSTM, механизма внимания и DeepSurv},
  school    = {НИУ ВШЭ, Факультет компьютерных наук},
  year      = {2026},
  type      = {Выпускная квалификационная работа}
}
```

---

## Лицензия

MIT — см. [LICENSE](LICENSE)
