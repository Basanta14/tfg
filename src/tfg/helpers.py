"""Steady stretches from one analyzer log"""

from pathlib import Path

import numpy as np
import pandas as pd

HZ_MIN = 45.2
HZ_MAX = 64.8
RAW_RATIO_MIN = 0.12
RAW_RATIO_MAX = 0.19
SPAN_HZ = 1.0
MIN_READINGS = 30
VOLTAGE_STD_MAX = 20.0
CURRENT_STD_MAX = 30.0
IQR_K = 1.5

_POINT_COLUMNS = ["f", "U", "I", "n"]
_EXTRA_COLUMNS = ("P", "Q", "FDP")


def _read_analyzer_csv(path: Path | str) -> pd.DataFrame:
    """CSV processing"""
    frame = pd.read_csv(path, sep=";", dtype=str)
    frame.columns = frame.columns.str.strip()
    for column in frame.columns.drop("Fecha_Hora", errors="ignore"):
        text = frame[column].astype(str).str.replace(",", ".", regex=False)
        frame[column] = pd.to_numeric(text, errors="coerce")
    return frame


def read_log(path: Path | str) -> tuple[pd.DataFrame, int]:
    """Rows inside the sweep, and how many were dropped."""
    frame = _read_analyzer_csv(path)
    frequency_hz = frame["F"] / 100.0
    raw_ratio = frame["U"] / frame["F"].replace(0, np.nan)
    current = frame["I"]
    valid = (
        frequency_hz.between(HZ_MIN, HZ_MAX)
        & raw_ratio.between(RAW_RATIO_MIN, RAW_RATIO_MAX)
        & frame["U"].notna()
        & current.notna()
        & (current >= 0)
    )
    mask = valid.to_numpy()
    readings = pd.DataFrame(
        {
            "f": frequency_hz.to_numpy()[mask],
            "U": frame["U"].to_numpy()[mask],
            "I": current.to_numpy()[mask],
        }
    )
    for name in _EXTRA_COLUMNS:
        if name in frame.columns:
            readings[name] = frame[name].to_numpy()[mask]
        else:
            readings[name] = np.nan
    return readings, int((~valid).sum())


def inliers(values: np.ndarray, k: float = IQR_K) -> np.ndarray:
    """True inside the IQR fence. A flat stretch keeps only the median."""
    q1, q3 = np.nanquantile(values, [0.25, 0.75])
    iqr = q3 - q1
    if iqr == 0:
        return values == np.nanmedian(values)
    fence = k * iqr
    return (values >= q1 - fence) & (values <= q3 + fence)


def frequency_stretches(frequency: np.ndarray):
    """[start, end) per dwell. A short run slides one sample forward."""
    count = len(frequency)
    start = 0
    while start < count:
        end = start
        low = high = frequency[start]
        while end + 1 < count:
            nxt = frequency[end + 1]
            new_low = min(low, nxt)
            new_high = max(high, nxt)
            if new_high - new_low > SPAN_HZ:
                break
            low, high = new_low, new_high
            end += 1
        if end - start + 1 >= MIN_READINGS:
            yield start, end + 1
            start = end + 1
        else:
            start += 1


def _steady_point(stretch: pd.DataFrame) -> dict | None:
    """Mean of one dwell, or nothing if it is too short or shaky."""
    voltage = stretch["U"].to_numpy()
    current = stretch["I"].to_numpy()
    kept = stretch[inliers(voltage) & inliers(current)]
    if len(kept) < MIN_READINGS:
        return None
    if float(kept["U"].std(ddof=0)) > VOLTAGE_STD_MAX:
        return None
    if float(kept["I"].std(ddof=0)) > CURRENT_STD_MAX:
        return None
    return {
        "f": float(kept["f"].mean()),
        "U": float(kept["U"].mean()),
        "I": float(kept["I"].mean()),
        "n": int(len(kept)),
    }


def operating_points(log_path: Path | str) -> pd.DataFrame:
    """One row per steady stretch."""
    log_path = Path(log_path)
    if not log_path.is_file():
        raise SystemExit(f"File not found: {log_path}")
    try:
        clean, _dropped = read_log(log_path)
    except (KeyError, ValueError, pd.errors.ParserError) as exc:
        raise SystemExit(
            f"{log_path.name} is not an analyzer log "
            "(F, U and I are required, separated by semicolons)."
        ) from exc

    if clean.empty:
        return pd.DataFrame(columns=_POINT_COLUMNS)

    rows = []
    for start, end in frequency_stretches(clean["f"].to_numpy()):
        point = _steady_point(clean.iloc[start:end])
        if point is not None:
            rows.append(point)
    return pd.DataFrame(rows, columns=_POINT_COLUMNS)
