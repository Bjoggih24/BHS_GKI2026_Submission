"""
Feature building utilities for hot water demand forecasting.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


HISTORY_LENGTH = 672
HORIZON = 72
BASE_TEMP = 17.0

WEATHER_NUMERIC_COLS = [
    "temperature",
    "windspeed",
    "cloud_coverage",
    "dewpoint",
    "rain_accumulated",
]

WEATHER_OBS_COLS = ["t", "f", "r", "td", "rh", "p", "fg"]


def _normalize_hourly_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Normalize timestamps to hourly, UTC, tz-naive for safe alignment."""
    out = pd.DatetimeIndex(idx)
    if out.tz is not None:
        out = out.tz_convert("UTC").tz_localize(None)
    out = out.floor("h")
    return out


@dataclass(frozen=True)
class FeatureConfig:
    time_features: Tuple[str, ...]
    history_features: Tuple[str, ...]
    weather_features: Tuple[str, ...]

    @property
    def names(self) -> Tuple[str, ...]:
        return self.time_features + self.history_features + self.weather_features


DEFAULT_FEATURE_CONFIG = FeatureConfig(
    time_features=(
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "doy_sin",
        "doy_cos",
    ),
    history_features=(
        "last_val",
        "lag_1",
        "lag_24",
        "lag_48",
        "lag_72",
        "lag_168",
        "lag_336",
        "lag_504",
        "lag_672",
        "roll24_mean",
        "roll24_std",
        "roll168_mean",
        "roll168_std",
        "anchor_daily_mean",
        "anchor_weekly_mean",
        "delta_last_roll24",
        "delta_last_roll168",
        "slope_24",
        "slope_168",
        "history_nan_frac",
    ),
    weather_features=tuple(
        [f"w_{c}" for c in WEATHER_NUMERIC_COLS]
        + [f"wo_last_{c}" for c in WEATHER_OBS_COLS]
        + [f"wo_mean24_{c}" for c in WEATHER_OBS_COLS]
        + [f"wo_mean168_{c}" for c in WEATHER_OBS_COLS]
    ),
)


def _build_feature_config_v2(include_total_context: bool) -> FeatureConfig:
    weather_names = (
        [f"w_{c}" for c in WEATHER_NUMERIC_COLS]
        + [
            "w_lead_hours",
            "w_issue_age_hours",
            "w_issue_hour_sin",
            "w_issue_hour_cos",
            "w_cov",
            "w_hdh",
            "w_wind_x_hdh",
        ]
        + [f"wo_last_{c}" for c in WEATHER_OBS_COLS]
        + [f"wo_mean24_{c}" for c in WEATHER_OBS_COLS]
        + [f"wo_mean168_{c}" for c in WEATHER_OBS_COLS]
        + [
            "wo_hdh24",
            "wo_hdh72",
            "wo_hdh168",
            "wo_temp_trend24",
            "wo_temp_trend168",
            "wo_wind_x_hdh24",
        ]
        + ["horizon_id"]
    )
    history_names = list(DEFAULT_FEATURE_CONFIG.history_features)
    if include_total_context:
        history_names += [f"tot_{name}" for name in DEFAULT_FEATURE_CONFIG.history_features]
    return FeatureConfig(
        time_features=DEFAULT_FEATURE_CONFIG.time_features,
        history_features=tuple(history_names),
        weather_features=tuple(weather_names),
    )


DEFAULT_FEATURE_CONFIG_V2 = _build_feature_config_v2(include_total_context=True)
DEFAULT_FEATURE_CONFIG_V2_NOTOTAL = _build_feature_config_v2(include_total_context=False)


def _is_v2_config(config: FeatureConfig) -> bool:
    return config.names in {
        DEFAULT_FEATURE_CONFIG_V2.names,
        DEFAULT_FEATURE_CONFIG_V2_NOTOTAL.names,
    }


def _to_datetime(ts: str) -> pd.Timestamp:
    return pd.to_datetime(ts, errors="coerce")


def build_horizon_times(forecast_start: str) -> pd.DatetimeIndex:
    """Return horizon timestamps for the next 72 hours."""
    start = _to_datetime(forecast_start)
    return pd.date_range(start=start, periods=HORIZON, freq="h")


def build_time_features(horizon_times: pd.DatetimeIndex) -> np.ndarray:
    """Cyclical time features from horizon timestamps."""
    hour = horizon_times.hour.to_numpy()
    dow = horizon_times.dayofweek.to_numpy()
    doy = horizon_times.dayofyear.to_numpy()

    hour_sin = np.sin(2 * np.pi * hour / 24.0)
    hour_cos = np.cos(2 * np.pi * hour / 24.0)
    dow_sin = np.sin(2 * np.pi * dow / 7.0)
    dow_cos = np.cos(2 * np.pi * dow / 7.0)
    doy_sin = np.sin(2 * np.pi * doy / 365.25)
    doy_cos = np.cos(2 * np.pi * doy / 365.25)

    return np.vstack([hour_sin, hour_cos, dow_sin, dow_cos, doy_sin, doy_cos]).T


def build_history_features(sensor_history_1d: np.ndarray) -> np.ndarray:
    """History-derived features for a single sensor, per horizon."""
    if sensor_history_1d.shape[0] != HISTORY_LENGTH:
        raise ValueError("sensor_history_1d must have length 672.")

    last_val = sensor_history_1d[-1]
    roll24 = sensor_history_1d[-24:]
    roll168 = sensor_history_1d[-168:]

    roll24_mean = np.nanmean(roll24)
    roll24_std = np.nanstd(roll24)
    roll168_mean = np.nanmean(roll168)
    roll168_std = np.nanstd(roll168)
    history_nan_frac = float(np.mean(np.isnan(sensor_history_1d)))

    def _slope(vals: np.ndarray) -> float:
        mask = ~np.isnan(vals)
        if mask.sum() < 2:
            return 0.0
        x = np.arange(vals.size)[mask]
        y = vals[mask]
        x_mean = x.mean()
        y_mean = y.mean()
        denom = np.sum((x - x_mean) ** 2)
        if denom < 1e-9:
            return 0.0
        return float(np.sum((x - x_mean) * (y - y_mean)) / denom)

    slope_24 = _slope(roll24)
    slope_168 = _slope(roll168)
    last_val_safe = np.nan_to_num(last_val, nan=roll24_mean if np.isfinite(roll24_mean) else 0.0)

    feats = []
    for h in range(HORIZON):
        # Lags relative to forecast time t+h, only if lag > h (within history window)
        def _lag_value(lag: int) -> float:
            if lag <= h:
                return np.nan
            idx = HISTORY_LENGTH - lag + h
            return sensor_history_1d[idx]

        lag_1 = _lag_value(1)
        lag_24 = _lag_value(24)
        lag_48 = _lag_value(48)
        lag_72 = _lag_value(72)
        lag_168 = _lag_value(168)
        lag_336 = _lag_value(336)
        lag_504 = _lag_value(504)
        lag_672 = _lag_value(672)

        # Seasonal anchors: daily and weekly mean over aligned lags if available
        daily_vals = [lag_24, lag_48, lag_72]
        weekly_vals = [lag_168, lag_336, lag_504, lag_672]
        anchor_daily_mean = np.nanmean(daily_vals) if np.any(~np.isnan(daily_vals)) else np.nan
        anchor_weekly_mean = np.nanmean(weekly_vals) if np.any(~np.isnan(weekly_vals)) else np.nan
        delta_last_roll24 = last_val_safe - roll24_mean
        delta_last_roll168 = last_val_safe - roll168_mean

        feats.append(
            [
                last_val,
                lag_1,
                lag_24,
                lag_48,
                lag_72,
                lag_168,
                lag_336,
                lag_504,
                lag_672,
                roll24_mean,
                roll24_std,
                roll168_mean,
                roll168_std,
                anchor_daily_mean,
                anchor_weekly_mean,
                delta_last_roll24,
                delta_last_roll168,
                slope_24,
                slope_168,
                history_nan_frac,
            ]
        )

    return np.array(feats, dtype=np.float32)


def load_weather_forecasts_agg(data_dir: Path) -> Optional[pd.DataFrame]:
    """Load and aggregate weather forecasts by value_date (mean across stations)."""
    candidates = [
        data_dir / "weather_forecasts.zip",
        data_dir / "weater_forecasts.zip",
        data_dir / "weather_forecasts.csv",
        data_dir / "weater_forecasts.csv",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        return None

    if path.suffix == ".zip":
        df = pd.read_csv(path, compression="zip")
    else:
        df = pd.read_csv(path)

    issue_col, valid_col = _infer_forecast_issue_valid_cols(df)
    ts = pd.to_datetime(df[valid_col], errors="coerce", utc=True)
    ts = ts.dt.floor("h").dt.tz_localize(None)
    df["valid_norm"] = ts
    df = df.dropna(subset=["valid_norm"])

    agg = df.groupby("valid_norm")[WEATHER_NUMERIC_COLS].mean().sort_index()
    agg = agg.resample("h").mean().interpolate("time")
    return agg


def _infer_forecast_issue_valid_cols(df: pd.DataFrame) -> tuple[str, str]:
    if "date_time" not in df.columns or "value_date" not in df.columns:
        raise ValueError("Forecast df missing required columns date_time/value_date")
    issue = pd.to_datetime(df["date_time"], errors="coerce", utc=True)
    valid = pd.to_datetime(df["value_date"], errors="coerce", utc=True)
    lead_a = (valid - issue) / pd.Timedelta(hours=1)
    if lead_a.median() < 0:
        return "value_date", "date_time"
    return "date_time", "value_date"


def load_weather_forecasts_runs(data_dir: Path) -> Optional[tuple[pd.DataFrame, pd.DatetimeIndex]]:
    """
    Load weather forecasts indexed by (issue_time, valid_time), averaged across stations.
    Returns (runs_df, issue_times).
    """
    candidates = [
        data_dir / "weather_forecasts.zip",
        data_dir / "weater_forecasts.zip",
        data_dir / "weather_forecasts.csv",
        data_dir / "weater_forecasts.csv",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        return None

    df = pd.read_csv(path, compression="zip" if path.suffix == ".zip" else None)
    issue_col, valid_col = _infer_forecast_issue_valid_cols(df)

    issue = pd.to_datetime(df[issue_col], errors="coerce", utc=True).dt.floor("h").dt.tz_localize(None)
    valid = pd.to_datetime(df[valid_col], errors="coerce", utc=True).dt.floor("h").dt.tz_localize(None)
    df["issue_norm"] = issue
    df["valid_norm"] = valid

    for c in WEATHER_NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    runs = df.groupby(["issue_norm", "valid_norm"])[WEATHER_NUMERIC_COLS].mean().sort_index()
    issue_times = runs.index.get_level_values(0).unique().sort_values()
    return runs, issue_times


def load_weather_observations_agg(data_dir: Path) -> Optional[pd.DataFrame]:
    """
    Load and aggregate weather observations by 'timi' (hourly mean across stations).
    Returns hourly tz-naive UTC index.
    """
    candidates = [
        data_dir / "weather_observations.zip",
        data_dir / "weater_observations.zip",
        data_dir / "weather_observations.csv",
        data_dir / "weater_observations.csv",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        return None

    usecols = ["timi", "t", "f", "r", "td", "rh", "p", "fg", "stod"]
    df = pd.read_csv(
        path,
        compression="zip" if path.suffix == ".zip" else None,
        usecols=lambda c: c in usecols,
        low_memory=False,
    )

    if "timi" not in df.columns:
        return None

    ts = pd.to_datetime(df["timi"], errors="coerce", utc=True)
    ts = ts.dt.floor("h").dt.tz_localize(None)
    df["timi_norm"] = ts
    df = df.dropna(subset=["timi_norm"])

    for c in WEATHER_OBS_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    agg_cols = [c for c in WEATHER_OBS_COLS if c in df.columns]
    if not agg_cols:
        return None

    agg = df.groupby("timi_norm")[agg_cols].mean().sort_index()
    agg = agg.resample("h").mean().interpolate("time")
    return agg


def build_weather_features(
    horizon_times: pd.DatetimeIndex,
    weather_forecast: Optional[np.ndarray],
    weather_agg: Optional[pd.DataFrame] = None,
) -> np.ndarray:
    """
    Build weather features for each horizon time.

    Priority:
      1) Use weather_forecast array from API if provided.
      2) Use aggregated weather forecasts DataFrame if provided.
      3) Return NaNs otherwise.
    """
    # If forecast is already numeric matrix aligned to horizon, return directly
    if weather_forecast is not None and getattr(weather_forecast, "ndim", 0) == 2:
        if weather_forecast.shape[1] == len(WEATHER_NUMERIC_COLS):
            return np.asarray(weather_forecast, dtype=np.float32)

    # Normalize horizon to hourly tz-naive UTC
    ht = _normalize_hourly_index(horizon_times)

    agg = None

    if weather_forecast is not None:
        try:
            df = pd.DataFrame(
                weather_forecast,
                columns=[
                    "date_time",
                    "station_id",
                    "temperature",
                    "windspeed",
                    "cloud_coverage",
                    "gust",
                    "humidity",
                    "winddirection",
                    "dewpoint",
                    "rain_accumulated",
                    "value_date",
                ],
            )

            issue_col, valid_col = _infer_forecast_issue_valid_cols(df)

            issue_norm = pd.to_datetime(df[issue_col], errors="coerce", utc=True).dt.floor("h").dt.tz_localize(None)
            valid_norm = pd.to_datetime(df[valid_col], errors="coerce", utc=True).dt.floor("h").dt.tz_localize(None)
            df["issue_norm"] = issue_norm
            df["valid_norm"] = valid_norm
            df = df.dropna(subset=["issue_norm", "valid_norm"])

            # Convert numeric columns from object to float (API sends mixed-type arrays)
            for col in WEATHER_NUMERIC_COLS:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            t0 = ht[0] if len(ht) > 0 else None
            if t0 is not None:
                issue_times = df["issue_norm"].unique()
                issue_times = pd.DatetimeIndex(issue_times).sort_values()
                k = issue_times.searchsorted(t0, side="right") - 1
                best = None
                best_run_df = None
                best_frac = -1.0
                for step in range(12):
                    idx = k - step
                    if idx < 0:
                        break
                    issue_star = issue_times[idx]
                    assert issue_star <= t0, f"LEAK: issue_star {issue_star} > t0 {t0}"
                    df_issue = df[df["issue_norm"] == issue_star]
                    run_df = df_issue.groupby("valid_norm")[WEATHER_NUMERIC_COLS].mean().sort_index()
                    run_df.index = _normalize_hourly_index(run_df.index)
                    aligned = run_df.reindex(ht)
                    frac = float(np.isfinite(aligned.to_numpy()).mean())
                    if frac > best_frac:
                        best_frac = frac
                        best = aligned
                        best_run_df = run_df
                    if frac >= 0.8:
                        break
                if best is None:
                    agg = None
                else:
                    if best_frac < 0.5 and best_run_df is not None:
                        aligned = best_run_df.reindex(ht, method="nearest", tolerance=pd.Timedelta("2h"))
                        best = aligned
                        best_frac = float(np.isfinite(best.to_numpy()).mean())
                    agg = None if best_frac < 0.5 else best
            else:
                agg = None
        except Exception:
            agg = None

    if agg is None:
        agg = weather_agg

    if agg is None:
        # Return zeros instead of NaN for robustness
        return np.zeros((HORIZON, len(WEATHER_NUMERIC_COLS)), dtype=np.float32)

    agg = agg.copy()
    agg.index = _normalize_hourly_index(agg.index)
    agg = agg.sort_index()

    aligned = agg.reindex(ht)

    if np.all(~np.isfinite(aligned.to_numpy())):
        aligned = agg.reindex(ht, method="nearest", tolerance=pd.Timedelta("59min"))

    # Impute remaining NaNs: ffill -> bfill -> column means -> zeros
    aligned = aligned.ffill().bfill()
    result = aligned.to_numpy(dtype=np.float32)
    
    # Final cleanup: replace any remaining NaNs with column means or zeros
    for j in range(result.shape[1]):
        col = result[:, j]
        mask = ~np.isfinite(col)
        if mask.any():
            col_mean = np.nanmean(col)
            if np.isfinite(col_mean):
                result[mask, j] = col_mean
            else:
                result[mask, j] = 0.0
    
    return result


def build_weather_features_v2(
    horizon_times: pd.DatetimeIndex,
    weather_forecast: Optional[np.ndarray],
) -> np.ndarray:
    """
    Build weather features with forecast meta for each horizon time.
    """
    # If forecast is already numeric matrix aligned to horizon, return with NaN meta
    if weather_forecast is not None and getattr(weather_forecast, "ndim", 0) == 2:
        if weather_forecast.shape[1] == len(WEATHER_NUMERIC_COLS):
            numeric = np.asarray(weather_forecast, dtype=np.float32)
            meta = _forecast_meta_features(_normalize_hourly_index(horizon_times), horizon_times[0], numeric)
            return np.hstack([numeric, meta])

    ht = _normalize_hourly_index(horizon_times)
    agg = None
    issue_star = None

    if weather_forecast is not None:
        try:
            df = pd.DataFrame(
                weather_forecast,
                columns=[
                    "date_time",
                    "station_id",
                    "temperature",
                    "windspeed",
                    "cloud_coverage",
                    "gust",
                    "humidity",
                    "winddirection",
                    "dewpoint",
                    "rain_accumulated",
                    "value_date",
                ],
            )
            issue_col, valid_col = _infer_forecast_issue_valid_cols(df)
            issue_norm = pd.to_datetime(df[issue_col], errors="coerce", utc=True).dt.floor("h").dt.tz_localize(None)
            valid_norm = pd.to_datetime(df[valid_col], errors="coerce", utc=True).dt.floor("h").dt.tz_localize(None)
            df["issue_norm"] = issue_norm
            df["valid_norm"] = valid_norm
            df = df.dropna(subset=["issue_norm", "valid_norm"])

            # Convert numeric columns from object to float (API sends mixed-type arrays)
            for col in WEATHER_NUMERIC_COLS:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            t0 = ht[0] if len(ht) > 0 else None
            if t0 is not None:
                issue_times = df["issue_norm"].unique()
                issue_times = pd.DatetimeIndex(issue_times).sort_values()
                k = issue_times.searchsorted(t0, side="right") - 1
                best = None
                best_run_df = None
                best_frac = -1.0
                best_issue = None
                for step in range(12):
                    idx = k - step
                    if idx < 0:
                        break
                    issue_star = issue_times[idx]
                    assert issue_star <= t0, f"LEAK: issue_star {issue_star} > t0 {t0}"
                    df_issue = df[df["issue_norm"] == issue_star]
                    run_df = df_issue.groupby("valid_norm")[WEATHER_NUMERIC_COLS].mean().sort_index()
                    run_df.index = _normalize_hourly_index(run_df.index)
                    aligned = run_df.reindex(ht)
                    frac = float(np.isfinite(aligned.to_numpy()).mean())
                    if frac > best_frac:
                        best_frac = frac
                        best = aligned
                        best_run_df = run_df
                        best_issue = issue_star
                    if frac >= 0.8:
                        break
                if best is not None:
                    if best_frac < 0.5 and best_run_df is not None:
                        aligned = best_run_df.reindex(ht, method="nearest", tolerance=pd.Timedelta("2h"))
                        best = aligned
                        best_frac = float(np.isfinite(best.to_numpy()).mean())
                    agg = None if best_frac < 0.5 else best
                    issue_star = best_issue
        except Exception:
            agg = None

    if agg is None:
        numeric = np.zeros((HORIZON, len(WEATHER_NUMERIC_COLS)), dtype=np.float32)
        meta = _forecast_meta_features(ht, issue_star, numeric)
        return np.hstack([numeric, meta])

    # Impute remaining NaNs: ffill -> bfill -> column means -> zeros
    agg = agg.ffill().bfill()
    numeric = agg.to_numpy(dtype=np.float32)
    
    # Final cleanup: replace any remaining NaNs with column means or zeros
    for j in range(numeric.shape[1]):
        col = numeric[:, j]
        mask = ~np.isfinite(col)
        if mask.any():
            col_mean = np.nanmean(col)
            if np.isfinite(col_mean):
                numeric[mask, j] = col_mean
            else:
                numeric[mask, j] = 0.0

    meta = _forecast_meta_features(ht, issue_star, numeric)
    return np.hstack([numeric, meta])


def _forecast_meta_features(
    horizon_times: pd.DatetimeIndex,
    issue_time: Optional[pd.Timestamp],
    numeric_matrix: np.ndarray,
) -> np.ndarray:
    if issue_time is None or len(horizon_times) == 0:
        meta = np.zeros((HORIZON, 7), dtype=np.float32)
        return meta
    issue_time = pd.Timestamp(issue_time)
    issue_hour = issue_time.hour
    issue_hour_sin = np.sin(2 * np.pi * issue_hour / 24.0)
    issue_hour_cos = np.cos(2 * np.pi * issue_hour / 24.0)
    issue_age = (pd.Timestamp(horizon_times[0]) - issue_time) / pd.Timedelta(hours=1)
    lead_hours = (horizon_times - issue_time) / pd.Timedelta(hours=1)
    lead_hours = np.asarray(lead_hours, dtype=np.float32)
    issue_age_hours = np.full(HORIZON, float(issue_age), dtype=np.float32)
    issue_hour_sin_arr = np.full(HORIZON, issue_hour_sin, dtype=np.float32)
    issue_hour_cos_arr = np.full(HORIZON, issue_hour_cos, dtype=np.float32)
    cov = np.isfinite(numeric_matrix).all(axis=1).astype(np.float32)
    temp = numeric_matrix[:, WEATHER_NUMERIC_COLS.index("temperature")]
    wind = numeric_matrix[:, WEATHER_NUMERIC_COLS.index("windspeed")]
    hdh = np.maximum(0.0, BASE_TEMP - temp)
    wind_x_hdh = wind * hdh
    meta = np.column_stack(
        [
            lead_hours,
            issue_age_hours,
            issue_hour_sin_arr,
            issue_hour_cos_arr,
            cov,
            hdh.astype(np.float32),
            wind_x_hdh.astype(np.float32),
        ]
    ).astype(np.float32)
    return meta


def build_weather_features_asof(
    horizon_times: pd.DatetimeIndex,
    forecast_start: str,
    runs: Optional[pd.DataFrame],
    issue_times: Optional[pd.DatetimeIndex],
) -> np.ndarray:
    """
    Build non-leaky forecast features as-of a single issue time.
    """
    if runs is None or issue_times is None or len(issue_times) == 0:
        return np.zeros((HORIZON, len(WEATHER_NUMERIC_COLS)), dtype=np.float32)

    t0 = pd.to_datetime(forecast_start, errors="coerce", utc=True).tz_localize(None).floor("h")
    issue_times = pd.DatetimeIndex(issue_times)
    k = issue_times.searchsorted(t0, side="right") - 1
    if k < 0:
        return np.zeros((HORIZON, len(WEATHER_NUMERIC_COLS)), dtype=np.float32)

    ht = _normalize_hourly_index(horizon_times)
    best = None
    best_run_df = None
    best_frac = -1.0
    for step in range(12):
        idx = k - step
        if idx < 0:
            break
        issue_star = issue_times[idx]
        assert issue_star <= t0, f"LEAK: issue_star {issue_star} > t0 {t0}"
        run_df = runs.xs(issue_star, level=0)
        run_df = run_df.copy()
        run_df.index = _normalize_hourly_index(run_df.index)
        aligned = run_df.reindex(ht)
        frac = float(np.isfinite(aligned.to_numpy()).mean())
        if frac > best_frac:
            best_frac = frac
            best = aligned
            best_run_df = run_df
        if frac >= 0.8:
            break

    if best is None:
        return np.zeros((HORIZON, len(WEATHER_NUMERIC_COLS)), dtype=np.float32)

    if best_frac < 0.5 and best_run_df is not None:
        aligned = best_run_df.reindex(ht, method="nearest", tolerance=pd.Timedelta("2h"))
        best = aligned
        best_frac = float(np.isfinite(best.to_numpy()).mean())

    if best_frac < 0.5:
        return np.zeros((HORIZON, len(WEATHER_NUMERIC_COLS)), dtype=np.float32)
    return best.to_numpy(dtype=np.float32)


def build_weather_features_asof_v2(
    horizon_times: pd.DatetimeIndex,
    forecast_start: str,
    runs: Optional[pd.DataFrame],
    issue_times: Optional[pd.DatetimeIndex],
) -> np.ndarray:
    if runs is None or issue_times is None or len(issue_times) == 0:
        return np.zeros((HORIZON, len(WEATHER_NUMERIC_COLS) + 7), dtype=np.float32)

    t0 = pd.to_datetime(forecast_start, errors="coerce", utc=True).tz_localize(None).floor("h")
    issue_times = pd.DatetimeIndex(issue_times)
    k = issue_times.searchsorted(t0, side="right") - 1
    if k < 0:
        return np.zeros((HORIZON, len(WEATHER_NUMERIC_COLS) + 7), dtype=np.float32)

    ht = _normalize_hourly_index(horizon_times)
    best = None
    best_run_df = None
    best_frac = -1.0
    best_issue = None
    for step in range(12):
        idx = k - step
        if idx < 0:
            break
        issue_star = issue_times[idx]
        assert issue_star <= t0, f"LEAK: issue_star {issue_star} > t0 {t0}"
        run_df = runs.xs(issue_star, level=0)
        run_df = run_df.copy()
        run_df.index = _normalize_hourly_index(run_df.index)
        aligned = run_df.reindex(ht)
        frac = float(np.isfinite(aligned.to_numpy()).mean())
        if frac > best_frac:
            best_frac = frac
            best = aligned
            best_run_df = run_df
            best_issue = issue_star
        if frac >= 0.8:
            break

    if best is None:
        return np.zeros((HORIZON, len(WEATHER_NUMERIC_COLS) + 7), dtype=np.float32)

    if best_frac < 0.5 and best_run_df is not None:
        aligned = best_run_df.reindex(ht, method="nearest", tolerance=pd.Timedelta("2h"))
        best = aligned
        best_frac = float(np.isfinite(best.to_numpy()).mean())

    if best_frac < 0.5:
        return np.zeros((HORIZON, len(WEATHER_NUMERIC_COLS) + 7), dtype=np.float32)

    # Impute remaining NaNs: ffill -> bfill -> column means -> zeros
    best = best.ffill().bfill()
    numeric = best.to_numpy(dtype=np.float32)
    
    # Final cleanup: replace any remaining NaNs with column means or zeros
    for j in range(numeric.shape[1]):
        col = numeric[:, j]
        mask = ~np.isfinite(col)
        if mask.any():
            col_mean = np.nanmean(col)
            if np.isfinite(col_mean):
                numeric[mask, j] = col_mean
            else:
                numeric[mask, j] = 0.0

    meta = _forecast_meta_features(ht, best_issue, numeric)
    return np.hstack([numeric, meta])


def build_weather_obs_summary_features(
    forecast_start: str,
    weather_history: Optional[np.ndarray],
    obs_agg: Optional[pd.DataFrame] = None,
) -> np.ndarray:
    """
    Returns (72, n_obs_feats). Uses API weather_history if provided, else obs_agg fallback.
    Summary stats over the past window (last, mean24, mean168) for each obs variable.
    """
    t0 = pd.to_datetime(forecast_start, errors="coerce", utc=True).tz_localize(None)
    hist_index = pd.date_range(end=t0 - pd.Timedelta(hours=1), periods=HISTORY_LENGTH, freq="h")

    mat = None
    cols = WEATHER_OBS_COLS

    if weather_history is not None:
        if getattr(weather_history, "ndim", 0) == 2 and weather_history.shape[1] == len(WEATHER_OBS_COLS):
            mat = np.asarray(weather_history, dtype=np.float32)
            if mat.shape[0] != HISTORY_LENGTH:
                mat = mat[-HISTORY_LENGTH:]
            cols = WEATHER_OBS_COLS
        else:
            mat = None

    if mat is None and weather_history is not None:
        try:
            df = pd.DataFrame(weather_history)
            ts = pd.to_datetime(df.iloc[:, 1], errors="coerce", utc=True).dt.floor("h").dt.tz_localize(None)
            df["timi_norm"] = ts
            df = df.dropna(subset=["timi_norm"])

            mapping = {
                "f": 2,
                "fg": 3,
                "t": 7,
                "rh": 10,
                "td": 11,
                "p": 12,
                "r": 13,
            }
            for k, idx in mapping.items():
                df[k] = pd.to_numeric(df.iloc[:, idx], errors="coerce")

            agg = df.groupby("timi_norm")[list(mapping.keys())].mean().sort_index()

            # Force deterministic column order to match training
            agg = agg.reindex(columns=WEATHER_OBS_COLS)

            # Reindex to our expected time range and forward-fill missing recent hours
            agg = agg.reindex(hist_index).ffill().bfill()
            mat = agg.to_numpy(dtype=np.float32)
            cols = WEATHER_OBS_COLS
        except Exception:
            mat = None

    if mat is None and obs_agg is not None:
        aligned = obs_agg.reindex(hist_index).ffill().bfill()
        mat = aligned.to_numpy(dtype=np.float32)
        cols = list(obs_agg.columns)

    if mat is None:
        n_feats = len(cols) * 3
        # Return zeros instead of NaN for robustness
        return np.zeros((HORIZON, n_feats), dtype=np.float32)

    # Ensure mat has no NaN by filling with column means or zeros
    if not mat.flags.writeable:
        mat = np.array(mat, copy=True)
    col_means = np.nanmean(mat, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    for j in range(mat.shape[1]):
        mat[:, j] = np.where(np.isnan(mat[:, j]), col_means[j], mat[:, j])

    last = mat[-1]
    mean24 = np.nanmean(mat[-24:], axis=0)
    mean168 = np.nanmean(mat[-168:], axis=0)
    
    # Final NaN cleanup
    last = np.nan_to_num(last, nan=0.0)
    mean24 = np.nan_to_num(mean24, nan=0.0)
    mean168 = np.nan_to_num(mean168, nan=0.0)
    
    feats_1d = np.concatenate([last, mean24, mean168], axis=0).astype(np.float32)

    return np.repeat(feats_1d[None, :], HORIZON, axis=0)


def build_weather_obs_summary_features_v2(
    forecast_start: str,
    weather_history: Optional[np.ndarray],
    obs_agg: Optional[pd.DataFrame] = None,
) -> np.ndarray:
    base_feats = build_weather_obs_summary_features(forecast_start, weather_history, obs_agg=obs_agg)
    if base_feats.ndim != 2:
        return base_feats
    n_obs = base_feats.shape[1] // 3
    if n_obs * 3 != base_feats.shape[1]:
        # Return zeros instead of NaN for robustness
        extra = np.zeros((HORIZON, 6), dtype=np.float32)
        return np.hstack([base_feats, extra])

    last = base_feats[0, :n_obs]
    mean24 = base_feats[0, n_obs : 2 * n_obs]
    mean168 = base_feats[0, 2 * n_obs : 3 * n_obs]

    cols = list(obs_agg.columns) if obs_agg is not None else list(WEATHER_OBS_COLS)
    try:
        temp_idx = cols.index("t")
        wind_idx = cols.index("f")
    except ValueError:
        # Return zeros instead of NaN for robustness
        extra = np.zeros((HORIZON, 6), dtype=np.float32)
        return np.hstack([base_feats, extra])

    temp_last = last[temp_idx]
    temp_m24 = mean24[temp_idx]
    temp_m168 = mean168[temp_idx]
    wind_m24 = mean24[wind_idx]

    hdh24 = max(0.0, BASE_TEMP - temp_m24)
    hdh72 = max(0.0, BASE_TEMP - ((temp_last + temp_m24) / 2.0))
    hdh168 = max(0.0, BASE_TEMP - temp_m168)
    temp_trend24 = temp_last - temp_m24
    temp_trend168 = temp_last - temp_m168
    wind_x_hdh24 = wind_m24 * hdh24

    extra_1d = np.array(
        [hdh24, hdh72, hdh168, temp_trend24, temp_trend168, wind_x_hdh24],
        dtype=np.float32,
    )
    extra = np.repeat(extra_1d[None, :], HORIZON, axis=0)
    return np.hstack([base_feats, extra])


def build_features_for_sensor(
    sensor_history_1d: np.ndarray,
    forecast_start: str,
    weather_forecast: Optional[np.ndarray] = None,
    weather_agg: Optional[pd.DataFrame] = None,
    weather_history: Optional[np.ndarray] = None,
    obs_agg: Optional[pd.DataFrame] = None,
    total_history_1d: Optional[np.ndarray] = None,
    runs: Optional[pd.DataFrame] = None,
    issue_times: Optional[pd.DatetimeIndex] = None,
    config: FeatureConfig = DEFAULT_FEATURE_CONFIG,
) -> np.ndarray:
    """Build full feature matrix (72, n_features) for one sensor."""
    horizon_times = build_horizon_times(forecast_start)
    time_feats = build_time_features(horizon_times)
    hist_feats = build_history_features(sensor_history_1d)
    if _is_v2_config(config):
        if weather_forecast is not None:
            weather_feats = build_weather_features_v2(horizon_times, weather_forecast)
        elif runs is not None and issue_times is not None:
            weather_feats = build_weather_features_asof_v2(horizon_times, forecast_start, runs, issue_times)
        else:
            weather_feats = build_weather_features(horizon_times, weather_forecast, weather_agg)
        obs_feats = build_weather_obs_summary_features_v2(forecast_start, weather_history, obs_agg=obs_agg)
        if total_history_1d is not None:
            tot_hist = build_history_features(total_history_1d)
        else:
            tot_hist = None
    else:
        weather_feats = build_weather_features(horizon_times, weather_forecast, weather_agg)
        obs_feats = build_weather_obs_summary_features(
            forecast_start, weather_history, obs_agg=obs_agg
        )
        tot_hist = None

    if tot_hist is not None:
        feats = np.hstack([time_feats, hist_feats, tot_hist, weather_feats, obs_feats])
    else:
        feats = np.hstack([time_feats, hist_feats, weather_feats, obs_feats])
    if _is_v2_config(config):
        horizon_id = np.arange(HORIZON, dtype=np.float32).reshape(-1, 1)
        feats = np.hstack([feats, horizon_id])
    if feats.shape[1] != len(config.names):
        raise ValueError("Feature dimension mismatch with config.")
    return feats

