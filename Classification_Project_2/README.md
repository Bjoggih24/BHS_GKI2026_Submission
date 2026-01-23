# Icelandic Habitat Classification (Project 2)

This README focuses on the reproducible pipeline used in `Classification_Project_2/`:
data prep, model training, ensembling, and API inference.

## Task Summary

Predict the habitat type for a 35x35 patch.

Input:
- 15 channels: 12 Sentinel-2 bands + elevation + slope + aspect.

Output:
- Integer class 0-70 (71 habitat types).

## Project Structure (Key Files)

API + Inference:
- `api.py` — FastAPI `/predict` endpoint.
- `model.py` — inference entry point; loads ensemble config and models.
- `pyproject.toml` — project configuration and dependencies.

Feature Engineering + Utilities:
- `feature_utils.py` — tabular feature engineering (v1/v2/v3).
- `utils.py` — patch encoding/decoding and data helpers.
- `config.py` — paths for data/artifacts/splits.

CNN:
- `cnn/dataset.py` — CNN dataset loading utilities.
- `cnn/model.py` — CNN model implementations.
- `cnn/ensemble_submit.json` — final ensemble configuration for API inference.

Training + Evaluation Scripts:
- `scripts/precompute_tabular_features.py` — build `X_tabular_v*.npy`.
- `scripts/train_tabular.py` — train ET/LGBM/XGB/MLP/Transformer tabular models.
- `scripts/train_cnn.py` — train CNN models (small_cnn and timm backbones).
- `scripts/run_overnight_v3_mix.py` — automated sweep (tabular + CNN).
- `scripts/tune_topk_ensemble.py` — coordinate-ascent ensemble tuning.
- `scripts/simple_ensemble_weights.py` — quick weighting tests.
- `scripts/eval_with_tta.py` — CNN TTA evaluation.

Artifacts (generated during training):
- `artifacts/features/` — precomputed tabular features (`.npy` files).
- `artifacts/experiments/` — trained model checkpoints and validation probabilities.
- `artifacts/split_seed42/` — train/val index splits (required).
- Other `artifacts/ensemble_*.json` files — alternative ensemble configurations.

## Final Results

**Validation Score (Leaderboard):** 89.9% accuracy (metric: 0.3899)

Submitted on: January 23, 2026 @ 06:05 AM

## Final Submission Ensemble

Active submission config:
- `cnn/ensemble_submit.json`

Models in the final ensemble:
- `ET_overfit_v3` (ExtraTreesClassifier, tabular)
- `LGBM_A_v3` (LightGBM, tabular)
- `cnn_small_2_nolog1p_seed5` (CNN, image-based)
- `MLP_TORCH_B` (PyTorch MLP, tabular)

The API loads `cnn/ensemble_submit.json` by default. To override:
```bash
export ENSEMBLE_CONFIG=path/to/custom_ensemble.json
```

**Note:** Model paths in the ensemble config are relative to the `artifacts/` directory and must be trained before use (see Reproducible Pipeline below).

## Data Requirements

Before running the pipeline, ensure you have:

1. **Split indices** in `artifacts/split_seed42/`:
   - `train_indices.npy` — indices for training set
   - `val_indices.npy` — indices for validation set

2. **Training data** in `data/train/`:
   - Raw satellite imagery patches (format depends on data loader)
   - Corresponding labels

3. **Configuration files** in `configs/`:
   - Already included; covers all model architectures and hyperparameters

## Reproducible Pipeline

### 1) Prepare data
```bash
python scripts/precompute_tabular_features.py --version v3
```

Outputs:
- `artifacts/features/X_tabular_v3.npy`
- `artifacts/features/y.npy`

### 2) Train tabular models (examples)
```bash
python scripts/train_tabular.py \
  --model extratrees \
  --params_json configs/overnight_v3/ET_overfit_v3.json \
  --features_path artifacts/features/X_tabular_v3.npy \
  --labels_path artifacts/features/y.npy \
  --out_dir artifacts/experiments/manual/ET_overfit_v3
```

Other key configs used:
- `configs/overnight_v3/LGBM_A_v3.json`
- `configs/tabular_targeted/MLP_TORCH_B.json`

### 3) Train CNN models
```bash
python scripts/train_cnn.py \
  --arch small_cnn \
  --params_json configs/overnight_v3/cnn_small_2_nolog1p_seed5.json \
  --out_dir artifacts/experiments/manual/cnn_small_2_nolog1p_seed5
```

### 4) Overnight sweep (optional)
```bash
python scripts/run_overnight_v3_mix.py
```

### 5) Ensemble tuning
Coordinate ascent (top models):
```bash
python scripts/tune_topk_ensemble.py \
  --prob_paths <val_probs...> \
  --names <names...> \
  --model_paths <model_paths...> \
  --types <types...> \
  --archs <archs...> \
  --feature_version v3 \
  --out_path artifacts/ensemble_topK.json
```

Quick weighting:
```bash
python scripts/simple_ensemble_weights.py --prob_paths ... --names ...
```

Final submission config is stored in:
- `cnn/ensemble_submit.json`

## Inference / API

Run locally:
```bash
python api.py
```

The endpoint returns a single class id per patch.

## Repro Checklist

- [ ] Data and split indices available in `data/` and `artifacts/split_seed42/`
- [ ] Precomputed tabular features (v3) in `artifacts/features/`
- [ ] Trained ET/LGBM/MLP tabular models in `artifacts/experiments/`
- [ ] Trained CNN model in `artifacts/experiments/`
- [ ] Ensemble config created and stored in `cnn/ensemble_submit.json`
- [ ] API tested with `python api.py`

## Notes

- Validation uses the fixed split in `artifacts/split_seed42/`.
- The primary performance lever is the ensemble configuration and model weights.
- Model paths in the ensemble config are relative to `artifacts/` and should be updated after training.
- Torch models (MLP, CNN) require `torch` and `torchvision` (optional dependencies in `pyproject.toml`).

## Experimental / Auxiliary Scripts

The following scripts were used during exploration but are not part of the final submission pipeline:

**Evaluation & Ensemble Tuning:**
- `eval_local.py`, `ensemble_val.py` — local validation on subsets of training data
- `tune_ensemble.py`, `tune_top3_ensemble.py` — earlier ensemble weight tuning (superseded by `tune_topk_ensemble.py`)
- `ensemble_from_probs.py` — combine pre-computed probability outputs
- `run_ensemble_sweep.py` — grid search over ensemble configurations

**Training Sweeps (alternative approaches):**
- `run_experiments.py`, `run_experiments_v2.py` — older experiment runners
- `run_cnn_sweep.py`, `run_tabular_sweep.py`, `run_tabular_custom_sweep.py` — hyperparameter sweeps for individual models
- `run_tabular_overnight.py`, `run_imgemb_tabular_sweep.py` — specialized sweeps (tabular only, with image embeddings)
- `resume_overnight_v3.py` — resume interrupted training runs

**Data & Feature Engineering:**
- `make_split.py` — create train/validation splits
- `merge_patches.py` — utilities for patch-level data manipulation
- `extract_img_embeddings.py` — extract pre-computed image features
- `compute_norm_stats.py` — compute normalization statistics for feature scaling
