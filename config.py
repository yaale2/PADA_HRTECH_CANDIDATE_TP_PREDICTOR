"""
Централизованная конфигурация проекта PADA_HRTECH_CANDIDATE_TP_PREDICTOR.

Все гиперпараметры вынесены сюда для воспроизводимости экспериментов.
Соответствует результатам: val C-index = 0.795 (95% CI [0.784; 0.806]).
"""
from pathlib import Path

# ── Пути ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).parent
DATA_DIR       = PROJECT_ROOT / "data"
MODEL_DIR      = PROJECT_ROOT / "models" / "retention"
FIG_DIR        = PROJECT_ROOT / "figures"
NOTEBOOKS_DIR  = PROJECT_ROOT / "notebooks"

RESUMES_JSONL  = DATA_DIR / "careerist_resumes.jsonl"

# Google Drive (для Colab)
DRIVE_BASE     = "/content/drive/MyDrive/careerist_retention"
USE_DRIVE      = True   # False — использовать локальные пути выше

# ── Параметры признаков ──────────────────────────────────────────────────────
TEXT_DIM       = 384    # размерность хэш-вектора текста эпизода
MAX_SEQ_LEN    = 12     # максимальная длина карьерной последовательности
NUMERIC_DIM    = 22     # число числовых статических признаков

# ── Архитектура модели ───────────────────────────────────────────────────────
HIDDEN_SIZE    = 128    # размер скрытого состояния BiLSTM (на направление)
NUM_LAYERS     = 2      # число слоёв LSTM
BIDIRECTIONAL  = True   # двунаправленный LSTM
DROPOUT        = 0.3    # dropout между слоями LSTM и в первом слое головки
MLP_HIDDEN     = 128    # размер первого скрытого слоя DeepSurv-головки

# ── Обучение ─────────────────────────────────────────────────────────────────
BATCH_SIZE          = 512
EPOCHS              = 100
LR                  = 3e-4
WEIGHT_DECAY        = 1e-4
GRAD_CLIP           = 5.0
WARMUP_EPOCHS       = 5
EARLY_STOP_PATIENCE = 15

# ── Разбиение выборки ────────────────────────────────────────────────────────
VAL_SHARE      = 0.2    # 80/20 стратифицированное разбиение
SEED           = 42

# ── Горизонты прогноза (месяцы) ──────────────────────────────────────────────
HORIZONS       = [6.0, 12.0, 24.0]

# ── Фильтрация данных ────────────────────────────────────────────────────────
MIN_LAST_JOB_DUR = 1.0  # минимальная длительность последнего эпизода (мес.)
