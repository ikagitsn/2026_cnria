"""Shared palette and daily aggregation used by the figure scripts.

Extracted so that the three regen_*.py scripts stand on their own and
the repository contains only what produces the paper.
"""

import pandas as pd

# One palette across every figure, so a variable keeps its colour from
# one panel to the next.
COLORS = {
    "T_int":        "#c0392b",   # brick red
    "T_ext":        "#2980b9",   # blue
    "HR_int":       "#16a085",   # teal
    "NH3":          "#8e44ad",   # purple
    "rainy":        "#27ae60",   # green — rainy season
    "dry":          "#d35400",   # burnt orange — dry season
    "transition":   "#7f8c8d",   # grey
    "comfort":      "#27ae60",   # comfort band
    "neutral":      "#34495e",
}


def daily_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """Daily min/max/mean and diurnal amplitudes.

    Days with fewer than 20 interior-temperature samples are dropped:
    a partial day would understate its own amplitude.
    """
    daily = df.resample("D").agg(
        T_min=("temperature", "min"),
        T_max=("temperature", "max"),
        T_mean=("temperature", "mean"),
        HR_min=("humidity", "min"),
        HR_max=("humidity", "max"),
        HR_mean=("humidity", "mean"),
        Text_min=("temperaturext", "min"),
        Text_max=("temperaturext", "max"),
        Text_mean=("temperaturext", "mean"),
        n=("temperature", "count"),
    )
    daily = daily[daily["n"] >= 20].copy()
    daily["T_amplitude"] = daily["T_max"] - daily["T_min"]
    daily["HR_amplitude"] = daily["HR_max"] - daily["HR_min"]
    daily["Text_amplitude"] = daily["Text_max"] - daily["Text_min"]
    return daily
