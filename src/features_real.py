"""
Feature engineering for last-job survival labels.

Каждое резюме рассматривается как один субъект.
- Метка: из ПОСЛЕДНЕГО по хронологии места работы
    duration = work_experience[-1].duration_months
    event    = 1 if NOT period.is_current else 0
- Признаки:
    * `sequence`: вся карьера (включая последнюю работу), но для текущей работы
      duration_months ЗАНУЛЯЕТСЯ + помечается dur_missing-флаг (защита от leakage).
    * `numeric`: статичные поля профиля + агрегаты по всей карьере (но `current_job_duration`
       и `total_experience_months` НЕ используются, потому что они напрямую содержат таргет).
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

TOKEN_RE = re.compile(r"[a-zа-яё0-9+#.]+", re.IGNORECASE)
# к каждому job-вектору добавляется 5 числовых: [log_dur, is_current_flag, position_norm, has_description, dur_missing]
JOB_NUMERIC_FEATURES = 5

NUMERIC_FEATURE_NAMES = [
    "salary_log",
    "salary_missing",
    "history_job_count",
    "history_avg_job_duration_months",
    "history_median_job_duration_months",
    "history_max_job_duration_months",
    "history_min_job_duration_months",
    "history_short_job_share_6m",
    "history_short_job_share_12m",
    "history_avg_gap_months",
    "history_max_gap_months",
    "history_career_span_months",
    "history_has_any",
    "education_count",
    "courses_count",
    "languages_count",
    "skills_count",
    "about_length_log",
    "relocation_restricted",
    "full_time_requested",
    "salary_per_history_experience_log",
    "has_higher_education",
]


@dataclass
class FeatureBatch:
    resume_hashes: List[str]
    sequence: List[List[List[float]]]
    lengths: List[int]
    numeric: List[List[float]]
    numeric_feature_names: List[str]
    text_dim: int
    max_seq_len: int
    durations: List[float]
    events: List[int]
    last_job_meta: List[Dict[str, Any]]

    @property
    def seq_input_dim(self) -> int:
        if not self.sequence or not self.sequence[0]:
            return self.text_dim + JOB_NUMERIC_FEATURES
        return len(self.sequence[0][0])


@dataclass
class NumericScaler:
    mean: List[float]
    scale: List[float]

    @classmethod
    def fit(cls, values: Sequence[Sequence[float]]) -> "NumericScaler":
        if not values:
            raise ValueError("Cannot fit scaler on empty matrix.")
        cols = list(zip(*values))
        mean = [sum(c) / len(c) for c in cols]
        scale = []
        for c, m in zip(cols, mean):
            var = sum((v - m) ** 2 for v in c) / len(c)
            std = math.sqrt(var)
            scale.append(std if std >= 1e-6 else 1.0)
        return cls(mean=mean, scale=scale)

    def transform(self, values: Sequence[Sequence[float]]) -> List[List[float]]:
        return [[(v - self.mean[i]) / self.scale[i] for i, v in enumerate(row)] for row in values]

    def to_dict(self) -> Dict[str, Any]:
        return {"mean": list(self.mean), "scale": list(self.scale)}


def load_resume_records(path: str | Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for ln, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{ln}: {exc}") from exc
    return records


def featurize_with_last_job_labels(
    records: Sequence[Dict[str, Any]],
    *,
    text_dim: int = 384,
    max_seq_len: int = 12,
    min_last_job_dur: float = 1.0,
) -> Tuple[FeatureBatch, Dict[str, int]]:
    """Извлечь признаки и метки last-job survival из реальных резюме."""
    if text_dim < 16:
        raise ValueError("text_dim must be >= 16.")
    if max_seq_len < 1:
        raise ValueError("max_seq_len must be >= 1.")

    seqs: List[List[List[float]]] = []
    lens: List[int] = []
    nums: List[List[float]] = []
    hashes: List[str] = []
    durations: List[float] = []
    events: List[int] = []
    last_meta: List[Dict[str, Any]] = []

    stats = {
        "total": 0,
        "no_work_experience": 0,
        "last_job_invalid_duration": 0,
        "last_job_missing_duration": 0,
        "kept": 0,
        "kept_event_1": 0,
        "kept_event_0": 0,
    }

    for row_index, rec in enumerate(records):
        stats["total"] += 1
        resume = rec.get("resume", {}) or {}
        source = rec.get("source", {}) or {}
        resume_hash = str(source.get("resume_hash") or f"row-{row_index}")

        jobs_all = _chronological_jobs(resume.get("work_experience") or [])
        if not jobs_all:
            stats["no_work_experience"] += 1
            continue

        last_job = jobs_all[-1]
        last_dur_raw = last_job.get("duration_months")
        if last_dur_raw is None or last_dur_raw == "":
            stats["last_job_missing_duration"] += 1
            continue
        last_dur = _safe_float(last_dur_raw)
        if last_dur < min_last_job_dur:
            stats["last_job_invalid_duration"] += 1
            continue

        last_period = last_job.get("period") or {}
        is_current = bool(last_period.get("is_current"))
        event = 0 if is_current else 1

        # Полная карьера в LSTM, обрезанная справа до max_seq_len (берём самые свежие)
        history_for_seq = jobs_all
        if len(history_for_seq) > max_seq_len:
            history_for_seq = history_for_seq[-max_seq_len:]

        # last job — это история без последней; для numeric features используем "историю до последней"
        history_for_numeric = jobs_all[:-1]

        seq_row = [[0.0 for _ in range(text_dim + JOB_NUMERIC_FEATURES)] for _ in range(max_seq_len)]
        last_in_seq_index = len(history_for_seq) - 1   # позиция последней работы в обрезанной последовательности
        for j_idx, job in enumerate(history_for_seq):
            mask_duration = (j_idx == last_in_seq_index) and is_current
            seq_row[j_idx] = _job_vector_label_safe(
                job, j_idx, max_seq_len, text_dim, mask_duration=mask_duration
            )

        seqs.append(seq_row)
        lens.append(max(1, len(history_for_seq)))
        nums.append(_numeric_features(resume, history_for_numeric))
        hashes.append(resume_hash)
        durations.append(float(last_dur))
        events.append(int(event))
        last_meta.append({
            "resume_hash": resume_hash,
            "last_position": last_job.get("position"),
            "last_industry": last_job.get("industry"),
            "is_current": is_current,
        })
        stats["kept"] += 1
        if event == 1:
            stats["kept_event_1"] += 1
        else:
            stats["kept_event_0"] += 1

    if not hashes:
        raise ValueError("После фильтрации не осталось ни одного резюме.")

    return (
        FeatureBatch(
            resume_hashes=hashes,
            sequence=seqs,
            lengths=lens,
            numeric=nums,
            numeric_feature_names=list(NUMERIC_FEATURE_NAMES),
            text_dim=text_dim,
            max_seq_len=max_seq_len,
            durations=durations,
            events=events,
            last_job_meta=last_meta,
        ),
        stats,
    )


def hashed_text_vector(text: str, dim: int = 384) -> List[float]:
    vector = [0.0 for _ in range(dim)]
    tokens = TOKEN_RE.findall((text or "").lower())
    if not tokens:
        return vector
    feats = list(tokens)
    feats.extend(f"{a}_{b}" for a, b in zip(tokens, tokens[1:]))
    for tok in feats:
        digest = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[bucket] += sign
    norm = math.sqrt(sum(v * v for v in vector))
    if norm > 0:
        vector = [v / norm for v in vector]
    return vector


def _job_vector_label_safe(
    job: Dict[str, Any],
    job_index: int,
    max_seq_len: int,
    text_dim: int,
    *,
    mask_duration: bool,
) -> List[float]:
    """
    Job vector for LSTM input.

    Если mask_duration=True (текущая работа, которая является таргетом),
    то duration_months зануляется и взводится dur_missing=1, чтобы не было target leakage.
    """
    description = job.get("description") or []
    if isinstance(description, list):
        desc_text = " ".join(str(x) for x in description[:8])
    else:
        desc_text = str(description)
    text = " ".join(str(p) for p in [job.get("position"), job.get("industry"), desc_text] if p)
    duration = _safe_float(job.get("duration_months"))
    period = job.get("period") or {}

    if mask_duration:
        log_dur_norm = 0.0
        dur_missing = 1.0
    else:
        log_dur_norm = math.log1p(duration) / math.log(241.0)
        dur_missing = 0.0

    return hashed_text_vector(text, text_dim) + [
        log_dur_norm,
        1.0 if period.get("is_current") else 0.0,
        float(job_index + 1) / float(max_seq_len),
        1.0 if desc_text.strip() else 0.0,
        dur_missing,
    ]


def _numeric_features(resume: Dict[str, Any], history: Sequence[Dict[str, Any]]) -> List[float]:
    """Численные признаки. Используем только историю ДО последней работы для агрегатов длительности,
       чтобы не было target leakage."""
    durations = [_safe_float(j.get("duration_months")) for j in history if _safe_float(j.get("duration_months")) > 0]
    if durations:
        avg_d = sum(durations) / len(durations)
        med_d = float(statistics.median(durations))
        max_d = max(durations)
        min_d = min(durations)
        short_6 = sum(1 for d in durations if d < 6) / len(durations)
        short_12 = sum(1 for d in durations if d < 12) / len(durations)
    else:
        avg_d = med_d = max_d = min_d = short_6 = short_12 = 0.0

    gaps, span = _gaps_and_span(history)
    salary = resume.get("salary") or {}
    salary_amount = _safe_float(salary.get("amount"))
    history_exp = sum(durations)
    about_len = len(str(resume.get("about") or ""))
    relocation = str(resume.get("relocation") or "").lower()
    employment = str(resume.get("employment_type") or "").lower()
    salary_per_exp = salary_amount / max(history_exp, 1.0)

    education = resume.get("education") or []
    has_higher = 0.0
    for ed in education:
        level = str(ed.get("level") or ed.get("name") or "").lower()
        if any(k in level for k in ("высш", "магистр", "бакалавр", "специалист", "higher", "master", "bachelor")):
            has_higher = 1.0
            break

    return [
        math.log1p(salary_amount),
        1.0 if salary_amount <= 0 else 0.0,
        float(len(history)),
        avg_d, med_d, max_d, min_d,
        short_6, short_12,
        (sum(gaps) / len(gaps)) if gaps else 0.0,
        max(gaps) if gaps else 0.0,
        span,
        1.0 if len(history) > 0 else 0.0,
        float(len(resume.get("education") or [])),
        float(len(resume.get("courses") or [])),
        float(len(resume.get("languages") or [])),
        float(len(resume.get("skills") or [])),
        math.log1p(about_len),
        1.0 if "невозмож" in relocation else 0.0,
        1.0 if "полная" in employment else 0.0,
        math.log1p(salary_per_exp),
        has_higher,
    ]


def _chronological_jobs(jobs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    indexed = list(enumerate(jobs))
    if not indexed:
        return []
    def key(item):
        idx, job = item
        period = job.get("period") or {}
        start = _ym_to_month_index(period.get("start"))
        if start is None:
            return (10**9, -idx)
        return (start, idx)
    return [job for _, job in sorted(indexed, key=key)]


def _gaps_and_span(jobs: Sequence[Dict[str, Any]]) -> Tuple[List[float], float]:
    periods: List[Tuple[int, int]] = []
    for job in jobs:
        period = job.get("period") or {}
        start = _ym_to_month_index(period.get("start"))
        end = _ym_to_month_index(period.get("end"))
        dur = int(_safe_float(job.get("duration_months")))
        if start is None:
            continue
        if end is None:
            end = start + max(dur, 1)
        periods.append((start, max(end, start)))
    if not periods:
        return [], 0.0
    periods.sort()
    gaps: List[float] = []
    prev_end = periods[0][1]
    for s, e in periods[1:]:
        gaps.append(float(max(0, s - prev_end)))
        prev_end = max(prev_end, e)
    span = float(max(e for _, e in periods) - min(s for s, _ in periods))
    return gaps, span


def _ym_to_month_index(value: Any) -> Optional[int]:
    if value is None:
        return None
    m = re.match(r"^(\d{4})(?:-(\d{2}))?$", str(value))
    if not m:
        return None
    return int(m.group(1)) * 12 + int(m.group(2) or "1")


def _safe_float(value: Any) -> float:
    if value in (None, ""):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def stratified_split(events: Sequence[int], val_share: float, seed: int) -> Tuple[List[int], List[int]]:
    """Stratified train/val split по event_observed для сбалансированной валидации."""
    rng = random.Random(seed)
    idx_0 = [i for i, e in enumerate(events) if int(e) == 0]
    idx_1 = [i for i, e in enumerate(events) if int(e) == 1]
    rng.shuffle(idx_0); rng.shuffle(idx_1)
    n0_val = max(1, int(round(len(idx_0) * val_share)))
    n1_val = max(1, int(round(len(idx_1) * val_share)))
    val = idx_0[:n0_val] + idx_1[:n1_val]
    train = idx_0[n0_val:] + idx_1[n1_val:]
    rng.shuffle(train); rng.shuffle(val)
    return train, val


# need random for stratified_split
import random
