"""
Open-Meteo Historical Archive — wrapper around openmeteo-requests
=====================================================================
Official Open-Meteo client, built on FlatBuffers (zero-copy NumPy)
rather than JSON, which is markedly faster on long series.

HTTP cache via requests-cache (SQLite), automatic retry via retry-requests.

Requirements:
    pip install openmeteo-requests requests-cache retry-requests pandas numpy

Quick usage:
    from openmeteo_fetch import fetch_openmeteo_historical
    df = fetch_openmeteo_historical(
        latitude=14.7957198, longitude=-16.9674297, elevation=87.42,
        start_date="2025-07-16", end_date="2025-12-04",
    )

Reference: https://open-meteo.com/en/docs/historical-weather-api
SDK GitHub : https://github.com/open-meteo/python-requests
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

import openmeteo_requests
import requests_cache
from retry_requests import retry


OPENMETEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Hourly variables useful for building thermal modelling.
# NOTE: order matters. The API returns the variables in exactly the
# order requested, and they are then read back by index via Variables(i).
DEFAULT_VARIABLES = (
    "temperature_2m",            # degC, air at 2 m
    "relative_humidity_2m",      # %, RH at 2 m
    "dew_point_2m",              # degC, dew point
    "apparent_temperature",      # degC, apparent
    "shortwave_radiation",       # W/m2, global radiation
    "direct_radiation",          # W/m2, direct
    "diffuse_radiation",         # W/m2, diffuse
    "cloud_cover",               # %, total cloud cover
    "wind_speed_10m",            # m/s, wind at 10 m
    "precipitation",             # mm, precipitation
    "surface_pressure",          # hPa
)


def _make_client(cache_dir: Path, expire_after: int = -1):
    """Build an openmeteo-requests client with a persistent cache and retry.

    expire_after = -1 means a permanent cache, which is sound for
    reanalysis data: it does not change once published.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "openmeteo_http_cache"  # SQLite file

    cache_session = requests_cache.CachedSession(
        str(cache_path), expire_after=expire_after,
    )
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    return openmeteo_requests.Client(session=retry_session)


def fetch_openmeteo_historical(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    elevation: Optional[float] = None,
    variables: Iterable[str] = DEFAULT_VARIABLES,
    cache_dir: Path = Path("./data_cache"),
    timezone: str = "UTC",
    expire_after: int = -1,
) -> pd.DataFrame:
    """Fetch hourly Open-Meteo Archive (ERA5) data.

    Parameters
    ----------
    latitude, longitude : decimal WGS84 coordinates.
    start_date, end_date : "YYYY-MM-DD".
    elevation : elevation in m (optional; Open-Meteo adjusts variables).
    variables : hourly variable names (see DEFAULT_VARIABLES).
                Order is preserved for index-based access on return.
    cache_dir : directory for the SQLite HTTP cache.
    timezone : "UTC" recommended, to match the sensor timestamps.
    expire_after : cache lifetime (-1 = permanent, recommended for archive).

    Retour
    ------
    DataFrame indexed by a UTC DatetimeIndex, one column per variable.
    """
    variables = list(variables)
    client = _make_client(cache_dir, expire_after=expire_after)

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": variables,
        "timezone": timezone,
    }
    if elevation is not None:
        params["elevation"] = elevation

    print(f"[Open-Meteo] {OPENMETEO_ARCHIVE_URL}")
    print(f"             {start_date} -> {end_date} "
          f"@ ({latitude:.4f}, {longitude:.4f})")

    responses = client.weather_api(OPENMETEO_ARCHIVE_URL, params=params)
    response = responses[0]

    print(f"             Resolved : {response.Latitude():.4f}N, "
          f"{response.Longitude():.4f}E, "
          f"elevation {response.Elevation():.1f} m, "
          f"UTC offset {response.UtcOffsetSeconds()}s")

    hourly = response.Hourly()
    # Index temporel reconstruit depuis Time / TimeEnd / Interval (epoch UNIX)
    date_index = pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left",
    )

    # Variables in the requested order (FlatBuffers, zero-copy)
    data = {"time": date_index}
    for i, var_name in enumerate(variables):
        data[var_name] = hourly.Variables(i).ValuesAsNumpy()

    df = pd.DataFrame(data).set_index("time")
    print(f"             {len(df):,} hourly steps received, "
          f"{len(variables)} variables")

    return df


def validate_against_observations(
    om_df: pd.DataFrame,
    observed: pd.Series,
    om_var: str = "temperature_2m",
) -> dict:
    """Compare an Open-Meteo variable against co-located sensor readings.

    Aligns by time interpolation, then computes MAE, RMSE, bias and
    correlation. Use as a sanity check before relying on the data
    Open-Meteo en production.
    """
    obs = observed.dropna()
    if len(obs) < 10:
        return {"n": len(obs), "note": "Trop peu d'observations valides"}

    om_series = om_df[om_var].reindex(
        om_df.index.union(obs.index)
    ).interpolate("time").reindex(obs.index)

    valid = om_series.notna() & obs.notna()
    if valid.sum() < 10:
        return {"n": int(valid.sum()), "note": "Insufficient overlap"}

    diff = om_series[valid] - obs[valid]
    return {
        "n": int(valid.sum()),
        "MAE": float(np.abs(diff).mean()),
        "RMSE": float(np.sqrt((diff ** 2).mean())),
        "bias_mean": float(diff.mean()),
        "correlation": float(om_series[valid].corr(obs[valid])),
        "obs_mean": float(obs[valid].mean()),
        "om_mean": float(om_series[valid].mean()),
    }


if __name__ == "__main__":
    # Standalone test: runs only if openmeteo-requests is installed
    df = fetch_openmeteo_historical(
        latitude=14.7957198,
        longitude=-16.9674297,
        elevation=87.42,
        start_date="2025-07-16",
        end_date="2025-12-04",
    )
    print("\nPreview:")
    print(df.head())
    print("\nStatistiques :")
    print(df.describe().T[["mean", "std", "min", "max"]].round(2))
