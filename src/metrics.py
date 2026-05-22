from __future__ import annotations

import math
from bisect import bisect_right
from typing import Dict, Iterable, List


def concordance_index(durations: Iterable[float], risk_scores: Iterable[float], events: Iterable[float]) -> float:
    durations_arr = [float(value) for value in durations]
    risks_arr = [float(value) for value in risk_scores]
    events_arr = [float(value) for value in events]

    concordant = 0.0
    comparable = 0.0
    n = len(durations_arr)
    for i in range(n):
        if events_arr[i] <= 0:
            continue
        for j in range(n):
            if durations_arr[i] >= durations_arr[j]:
                continue
            comparable += 1.0
            if risks_arr[i] > risks_arr[j]:
                concordant += 1.0
            elif abs(risks_arr[i] - risks_arr[j]) < 1e-12:
                concordant += 0.5

    if comparable == 0:
        return float("nan")
    return float(concordant / comparable)


def breslow_baseline_cumulative_hazard(
    durations: Iterable[float],
    events: Iterable[float],
    log_risks: Iterable[float],
) -> Dict[str, List[float]]:
    durations_arr = [float(value) for value in durations]
    events_arr = [float(value) for value in events]
    risks_arr = [math.exp(float(value)) for value in log_risks]

    event_times = sorted(set(time for time, event in zip(durations_arr, events_arr) if event > 0))
    times: List[float] = []
    cumulative: List[float] = []
    total = 0.0
    for time in event_times:
        events_at_time = sum(1.0 for duration, event in zip(durations_arr, events_arr) if duration == time and event > 0)
        at_risk_sum = sum(risk for duration, risk in zip(durations_arr, risks_arr) if duration >= time)
        if at_risk_sum <= 0:
            continue
        total += events_at_time / at_risk_sum
        times.append(time)
        cumulative.append(total)
    return {"times": times, "cumulative_hazard": cumulative}


def survival_at_horizons(log_risk: float, baseline: Dict[str, List[float]], horizons: Iterable[float]) -> Dict[str, float]:
    times = [float(value) for value in baseline.get("times") or []]
    hazards = [float(value) for value in baseline.get("cumulative_hazard") or []]
    relative_risk = math.exp(float(log_risk))
    result: Dict[str, float] = {}
    for horizon in horizons:
        key = f"survival_{int(horizon)}m"
        if len(times) == 0:
            result[key] = float("nan")
            continue
        index = bisect_right(times, float(horizon)) - 1
        if index < 0:
            baseline_hazard = 0.0
        else:
            baseline_hazard = float(hazards[index])
        result[key] = math.exp(-baseline_hazard * relative_risk)
    return result
