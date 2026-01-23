#!/usr/bin/env python3
"""Evaluate different ensemble combinations."""
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

print("🚀 Starting ensemble evaluation...")
print(f"   Root: {ROOT}")

print("📦 Importing models_v2...")
from models_v2 import predict_lgbm_v2, predict_extratrees_v1
print("   ✓ models_v2 imported")

print("📦 Importing model.py...")
from model import _predict_lgbm_per_sensor_post2022_v1, _resolve
print("   ✓ model.py imported")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts" / "runs"

# Models to ensemble
MODELS = {
    "lgbm_new": ARTIFACTS_DIR / "20260123_015301_lgbm_ps_full_v2",
    "extratrees": ARTIFACTS_DIR / "20260123_015951_et_ps_full_v1",
    "lgbm_2024": ARTIFACTS_DIR / "20260123_021823_lgbm_ps_2024only_v1",
    "lgbm_noweather": ARTIFACTS_DIR / "20260123_022133_lgbm_ps_noweather_v1",
    "lgbm_old_best": ARTIFACTS_DIR / "20260121_124208_lgbm_ps_post2022_stride24",
}

# Ensemble configurations to test
ENSEMBLES = [
    # (name, {model: weight})
    ("lgbm_new_only", {"lgbm_new": 1.0}),
    ("extratrees_only", {"extratrees": 1.0}),
    ("lgbm_old_best_only", {"lgbm_old_best": 1.0}),
    ("lgbm_new + extratrees", {"lgbm_new": 0.5, "extratrees": 0.5}),
    ("lgbm_new + lgbm_old", {"lgbm_new": 0.5, "lgbm_old_best": 0.5}),
    ("lgbm_new + extratrees + lgbm_old", {"lgbm_new": 0.4, "extratrees": 0.3, "lgbm_old_best": 0.3}),
    ("all_4_equal", {"lgbm_new": 0.25, "extratrees": 0.25, "lgbm_old_best": 0.25, "lgbm_noweather": 0.25}),
    ("top3_weighted", {"lgbm_new": 0.5, "extratrees": 0.3, "lgbm_old_best": 0.2}),
]


def compute_skill(y_pred: np.ndarray, y_true: np.ndarray, y_baseline: np.ndarray) -> float:
    """Compute skill score vs baseline."""
    rmse_model = np.sqrt(np.mean((y_pred - y_true) ** 2))
    rmse_baseline = np.sqrt(np.mean((y_baseline - y_true) ** 2))
    if rmse_baseline < 1e-9:
        return np.nan
    return 1.0 - rmse_model / rmse_baseline


def baseline_lag72(sensor_history: np.ndarray) -> np.ndarray:
    """Simple lag-72 baseline."""
    return sensor_history[-72:, :]


def load_weather_agg():
    """Load pre-aggregated weather data."""
    weather_agg = pd.read_csv(DATA_DIR / "weather_agg.csv", parse_dates=True, index_col=0)
    obs_agg = pd.read_csv(DATA_DIR / "weather_obs_agg.csv", parse_dates=True, index_col=0)
    return weather_agg, obs_agg


def predict_model(model_name: str, run_dir: Path, sensor_history: np.ndarray, 
                  timestamp: str, weather_fcst, weather_obs, weather_agg, obs_agg) -> np.ndarray:
    """Get prediction from a single model."""
    # Check if it's an old format model (has models/ folder)
    if (run_dir / "models").exists():
        # Old format - use original model.py loader
        postproc_path = run_dir / "postproc.npz"
        return _predict_lgbm_per_sensor_post2022_v1(
            sensor_history, timestamp, weather_fcst, weather_obs,
            _resolve(run_dir), _resolve(postproc_path)
        )
    elif "extratrees" in model_name:
        return predict_extratrees_v1(
            sensor_history, timestamp, weather_fcst, weather_obs,
            run_dir, weather_agg=weather_agg, obs_agg=obs_agg
        )
    else:
        return predict_lgbm_v2(
            sensor_history, timestamp, weather_fcst, weather_obs,
            run_dir, weather_agg=weather_agg, obs_agg=obs_agg
        )


def main():
    total_start = time.time()
    
    # Load validation data
    print("\n" + "="*60)
    print("📂 [1/3] Loading validation data...")
    print("="*60)
    t0 = time.time()
    val_data = np.load(DATA_DIR / "val_clean.npz", allow_pickle=True)
    X = val_data["X"]
    y = val_data["y"]
    timestamps = val_data["timestamps"]
    weather_fcsts = val_data["weather_forecasts"]
    weather_obs = val_data["weather_observations"]
    print(f"   ✓ Loaded {len(X)} samples in {time.time()-t0:.1f}s")
    print(f"   X shape: {X.shape}, y shape: {y.shape}")
    
    # Load weather aggregates
    print("\n" + "="*60)
    print("🌤️  [2/3] Loading weather aggregates...")
    print("="*60)
    t0 = time.time()
    weather_agg, obs_agg = load_weather_agg()
    print(f"   ✓ Loaded in {time.time()-t0:.1f}s")
    print(f"   weather_agg: {len(weather_agg)} rows, obs_agg: {len(obs_agg)} rows")
    
    # Evaluate each ensemble
    print("\n" + "="*60)
    print("🎯 [3/3] Evaluating ensembles...")
    print("="*60)
    results = []
    
    for ens_idx, (ens_name, weights) in enumerate(ENSEMBLES):
        print(f"\n{'─'*60}")
        print(f"🔬 Ensemble {ens_idx+1}/{len(ENSEMBLES)}: {ens_name}")
        print(f"   Models: {list(weights.keys())}")
        print(f"   Weights: {list(weights.values())}")
        print(f"{'─'*60}")
        
        skills = []
        errors = 0
        ens_start = time.time()
        
        for i in range(len(X)):
            sample_start = time.time()
            sensor_history = X[i]
            y_true = y[i]
            ts = str(timestamps[i])
            wf = weather_fcsts[i] if weather_fcsts[i] is not None else None
            wo = weather_obs[i] if weather_obs[i] is not None else None
            
            # Get predictions from each model in ensemble
            preds = []
            model_weights = []
            for model_name, weight in weights.items():
                run_dir = MODELS[model_name]
                try:
                    pred = predict_model(model_name, run_dir, sensor_history, ts, wf, wo, weather_agg, obs_agg)
                    preds.append(pred)
                    model_weights.append(weight)
                except Exception as e:
                    print(f"   ❌ Error {model_name}: {e}")
                    errors += 1
                    continue
            
            if len(preds) == 0:
                continue
            
            # Weighted average
            model_weights = np.array(model_weights)
            model_weights /= model_weights.sum()  # normalize
            y_pred = sum(w * p for w, p in zip(model_weights, preds))
            
            # Compute skill
            y_base = baseline_lag72(sensor_history)
            skill = compute_skill(y_pred, y_true, y_base)
            skills.append(skill)
            
            # Progress output every 10 samples
            if (i + 1) % 10 == 0 or i == len(X) - 1:
                elapsed = time.time() - ens_start
                avg_time = elapsed / (i + 1)
                eta = avg_time * (len(X) - i - 1)
                running_mean = np.nanmean(skills) if skills else 0
                print(f"   [{i+1:3d}/{len(X)}] skill={skill:.3f} | running_mean={running_mean:.4f} | {avg_time:.2f}s/sample | ETA: {eta:.0f}s")
        
        skills = np.array(skills, dtype=np.float64)
        valid_skills = skills[np.isfinite(skills)]
        
        mean_skill = np.nanmean(skills)
        std_skill = np.nanstd(skills)
        neg_pct = (valid_skills < 0).mean() * 100 if len(valid_skills) > 0 else 0
        above_03 = (valid_skills > 0.3).mean() * 100 if len(valid_skills) > 0 else 0
        
        results.append({
            "name": ens_name,
            "mean": mean_skill,
            "std": std_skill,
            "neg%": neg_pct,
            ">0.3%": above_03,
        })
        
        ens_time = time.time() - ens_start
        print(f"\n   ✅ RESULT: Mean={mean_skill:.4f} ± {std_skill:.4f}")
        print(f"      Neg: {neg_pct:.1f}%, >0.3: {above_03:.1f}%, Errors: {errors}")
        print(f"      Time: {ens_time:.1f}s")
    
    # Summary
    total_time = time.time() - total_start
    print("\n\n" + "=" * 80)
    print("🏆 FINAL ENSEMBLE COMPARISON (sorted by mean skill)")
    print("=" * 80)
    print(f"{'Rank':<5} {'Ensemble':<35} {'Mean':>8} {'Std':>8} {'Neg%':>8} {'>0.3%':>8}")
    print("-" * 80)
    for rank, r in enumerate(sorted(results, key=lambda x: -x["mean"]), 1):
        emoji = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
        print(f"{emoji}{rank:<3} {r['name']:<35} {r['mean']:>8.4f} {r['std']:>8.4f} {r['neg%']:>7.1f}% {r['>0.3%']:>7.1f}%")
    print("=" * 80)
    print(f"⏱️  Total time: {total_time/60:.1f} minutes")
    print("\n🎉 Done! Best ensemble: {sorted(results, key=lambda x: -x['mean'])[0]['name']}")


if __name__ == "__main__":
    main()
