# Hierarchical Time Series Forecasting (Hot Water Demand)

This README documents the reproducible pipeline for hot water demand forecasting:
data preparation, model training, ensemble selection, and API inference.

## Final Results

**Test Score (Leaderboard):** Skill: 0.34

## Project Structure (Key Files)

API + Inference:
- `api.py` -FastAPI `/predict` endpoint for real-time forecasts.
- `model.py` -inference entry point; loads and applies ensemble from `artifacts/active_model.json`.
- `models_v2.py` -model loading/prediction utilities used by `model.py`.

Feature Engineering + Utilities:
- `feature_utils.py` -time/history/weather feature builders for training and inference.
- `utils.py` -data loaders, scoring metrics, baselines, time-based train/val splits.

Configuration:
- `requirements.txt` -Python dependencies.

Training + Evaluation:
- `scripts/build_train_full.py` -extract raw data and build `data/train_full.npz` windows.
- `scripts/build_weather_agg.py` -build `data/weather_agg.csv` and `data/weather_obs_agg.csv`.
- `scripts/train_models.py` -train main per-sensor models (LGBM, ExtraTrees).

## Final Model (Submitted)

The deployed ensemble is controlled by:
- `artifacts/active_model.json`

Ensemble composition (equal weights):
- `lgbm_ps_full_v2` (LightGBM, 300 estimators, weather-aware) in `artifacts/runs/20260123_015301_lgbm_ps_full_v2/`
- `et_ps_full_v1` (ExtraTrees, weather-aware) in `artifacts/runs/20260123_015951_et_ps_full_v1/`

Model configurations:
- `artifacts/runs/20260123_015301_lgbm_ps_full_v2/config.json` -LGBM hyperparameters
- `artifacts/runs/20260123_015951_et_ps_full_v1/config.json` -ExtraTrees hyperparameters

## Data Requirements

Before running the pipeline:

1. **Raw data files** (already in `data/` as zips):
   - `data/sensors.zip` -hourly sensor readings (2020–2024)
   - `data/weather_forecasts.zip` -weather forecasts (2022-06 onwards)
   - `data/weather_observations.zip` -historical weather observations

2. **After extraction**, the following CSVs are available:
   - `sensor_timeseries.csv` -45 columns (sensor readings)
   - `weather_forecasts.csv` -per-station forecasts
   - `weather_observations.csv` -per-station observations

## Reproducible Pipeline

### 0) Extract raw data files

```bash
cd data
unzip sensors.zip
unzip weather_forecasts.zip
unzip weather_observations.zip
cd ..
```

This creates:
- `data/sensor_timeseries.csv`
- `data/weather_forecasts.csv`
- `data/weather_observations.csv`

### 1) Build weather aggregates

```bash
python scripts/build_weather_agg.py
```

This creates:
- `data/weather_agg.csv`
- `data/weather_obs_agg.csv`

### 2) Build training windows

```bash
python scripts/build_train_full.py \
  --stride 24 \
  --min-date 2022-07-01 \
  --output data/train_full.npz
```

Notes:
- Produces `(X, y, timestamps)` with 672-hour (28-day) history and 72-hour (3-day) forecast horizon
- Starts from 2022-07-01 when weather forecasts become available
- Weather data loaded on-demand during training (not stored in NPZ for size efficiency)

### 3) Train the main models

```bash
python scripts/train_models.py --train-path data/train_full.npz --models lgbm,extratrees
```

This trains:
- `lgbm_ps_full_v2` (LightGBM with weather, best single model)
- `et_ps_full_v1` (ExtraTrees with weather, for ensemble diversity)

Outputs are written to:
- `artifacts/runs/<timestamp>_<model_name>/` (contains model files and config.json)

### 4) Update the active model config (if needed)

The ensemble is configured in `artifacts/active_model.json`:
```json
{
  "type": "ensemble",
  "models": [
    {"type": "lgbm_v2", "run_dir": "artifacts/runs/20260123_015301_lgbm_ps_full_v2"},
    {"type": "extratrees_v1", "run_dir": "artifacts/runs/20260123_015951_et_ps_full_v1"}
  ],
  "weights": [0.5, 0.5],
  "reconcile_beta": 0.0,
  "total_idx": 44
}
```

### 5) Run the API

```bash
python api.py
```

Server starts on `http://0.0.0.0:8080`

**Input format** (matches competition API):
- `sensor_history`: array of shape (672, 45) -hourly readings for all sensors
- `timestamp`: ISO 8601 string -forecast target time
- `weather_forecast`: rows with (date_time, station_id, temperature, windspeed, cloud_coverage, gust, humidity, winddirection, dewpoint, rain_accumulated, value_date)
- `weather_history`: rows with historical weather observations for context

**Output**: JSON with 72-hour forecasts for all 45 sensors

## Repro Checklist

- [ ] Raw data extracted from zips in `data/`
- [ ] Weather aggregates built with `scripts/build_weather_agg.py`
- [ ] `data/train_full.npz` built with `scripts/build_train_full.py`
- [ ] Both models trained with `scripts/train_models.py --models lgbm,extratrees`
- [ ] `artifacts/active_model.json` configured with desired ensemble
- [ ] API running successfully with `python api.py`

## Notes

- Weather data is loaded on-demand during training and inference (not pre-cached)
- Ensemble uses equal weights (0.5, 0.5) for LGBM and ExtraTrees
- The `reconcile_beta` parameter in `active_model.json` controls hierarchical reconciliation (currently 0.0)
