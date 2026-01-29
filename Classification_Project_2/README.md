# Icelandic Habitat Classification (Project 2)

This README focuses on the reproducible pipeline used in `Classification_Project_2/`.
It covers data prep, model training, ensembling, and API inference.

## Task Summary

Predict the habitat type for a 35x35 patch.

Input:
- 15 channels: 12 Sentinel-2 bands + elevation + slope + aspect.

Output:
- Integer class 0-70 (71 habitat types).

## Project Structure (Key Files)

API + Inference:
- `api.py` - FastAPI `/predict` endpoint.
- `model.py` - Inference entry point; loads the ensemble config and models.
- `pyproject.toml` - Project configuration and dependencies.

Feature Engineering + Utilities:
- `feature_utils.py` - Tabular feature engineering (v1, v2, v3).
- `utils.py` - Patch encoding and decoding plus data helpers.
- `config.py` - Paths for data, artifacts, and splits.

CNN:
- `cnn/dataset.py` - CNN dataset loading utilities.
- `cnn/model.py` - CNN model implementations.
- `cnn/ensemble_submit.json` - Final ensemble configuration for API inference.

Training + Evaluation Scripts:
- `scripts/precompute_tabular_features.py` - Build `X_tabular_v*.npy`.
- `scripts/train_tabular.py` - Train ET, LGBM, XGB, MLP, and transformer tabular models.
- `scripts/train_cnn.py` - Train CNN models (small_cnn and timm backbones).
- `scripts/simple_ensemble_weights.py` - Quick weighting tests.

Artifacts (generated during training):
- `artifacts/features/` - Precomputed tabular features (`.npy` files).
- `artifacts/experiments/` - Trained model checkpoints and validation probabilities.
- `artifacts/split_seed42/` - Train and val index splits (required).
- Other `artifacts/ensemble_*.json` files - Alternative ensemble configurations.

## Final Results

**Validation Score (Leaderboard):** Weighted F1: 0.3899

**Test Score (Leaderboard):** Weighted F1: 0.39

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

**Note:** `ENSEMBLE_CONFIG` can be absolute or relative to the project root. Model paths inside the ensemble config are relative to the `artifacts/` directory and must be trained before use (see Reproducible Pipeline below). All tabular models in the final ensemble use **v3 features**.

## Data Requirements

Before running the pipeline, ensure you have:

1. **Split indices** in `artifacts/split_seed42/`:
   - `train_idx.npy` -indices for training set
   - `val_idx.npy` -indices for validation set

2. **Raw training data zips** in `data/`:
   - `train_data_part1.zip`
   - `train_data_part2.zip`

3. **Training data** in `data/train/` (created by `scripts/prepare_train_data.py`):
   - `patches.npy` - Raw satellite imagery patches
   - `train.csv` - Labels (column: `vistgerd_idx`)
   - `patches_part1.npy`, `patches_part2.npy` - Intermediate files from the zip extract

4. **Submission configs** in `submission_configs/`:
   - Only the configs used by the final ensemble

## Reproducible Pipeline

### 0) Create a virtual environment + install deps
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 1) Prepare data + splits
```bash
python scripts/prepare_train_data.py
python scripts/make_split.py
python scripts/precompute_tabular_features.py --version v3
```

Outputs:
- `artifacts/features/X_tabular_v3.npy`
- `artifacts/features/y.npy`
- `artifacts/split_seed42/train_idx.npy`
- `artifacts/split_seed42/val_idx.npy`

### 2) Train tabular models (examples)
```bash
python scripts/train_tabular.py \
  --model extratrees \
  --params_json submission_configs/ET_overfit_v3.json \
  --features_path artifacts/features/X_tabular_v3.npy \
  --labels_path artifacts/features/y.npy \
  --out_dir artifacts/experiments/manual/ET_overfit_v3
```

Other key configs used:
- `submission_configs/LGBM_A_v3.json`
- `submission_configs/MLP_TORCH_B.json`
All tabular models in the final ensemble should use `artifacts/features/X_tabular_v3.npy`.

### 3) Train CNN models
```bash
python scripts/train_cnn.py \
  --arch small_cnn \
  --params_json submission_configs/cnn_small_2_nolog1p_seed5.json \
  --out_dir artifacts/experiments/manual/cnn_small_2_nolog1p_seed5
```

### 4) Ensemble tuning
Greedy selection + coord ascent (validation probs):
```bash
python scripts/tune_ensemble.py \
  --prob_paths \
    artifacts/experiments/manual/ET_overfit_v3/val_probs.npy \
    artifacts/experiments/manual/LGBM_A_v3/val_probs.npy \
    artifacts/experiments/manual/cnn_small_2_nolog1p_seed5/val_probs.npy \
    artifacts/experiments/manual/MLP_TORCH_B/val_probs.npy \
  --names ET_overfit_v3 LGBM_A_v3 cnn_small_2_nolog1p_seed5 MLP_TORCH_B \
  --model_types tabular tabular cnn tabular \
  --model_paths \
    experiments/manual/ET_overfit_v3/model.joblib \
    experiments/manual/LGBM_A_v3/model.joblib \
    experiments/manual/cnn_small_2_nolog1p_seed5/model.pt \
    experiments/manual/MLP_TORCH_B/model.pt \
  --cnn_archs "" "" small_cnn "" \
  --cnn_norm_paths "" "" experiments/manual/cnn_small_2_nolog1p_seed5/norm.npz "" \
  --force_all \
  --out_json artifacts/ensemble_submit.json
```

Final submission config is stored in:
- `cnn/ensemble_submit.json` (or set `ENSEMBLE_CONFIG=artifacts/ensemble_submit.json`)

Important: `cnn/ensemble_submit.json` in this repo points to model files under `artifacts/` that are not tracked. After you train your own models, either update this file or export a new config from `scripts/tune_ensemble.py`.

## Inference / API

Run locally:
```bash
python api.py
```

The endpoint returns a single class id per patch. The API requires trained artifacts and a valid ensemble config.

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
- Torch models (MLP, CNN) require `torch` and `torchvision`. The API also imports torch at startup.

## Git Tracking

Project code, configs, and small metadata are tracked in git. The following generated artifacts are excluded (regenerate via the pipeline):

- `artifacts/experiments/` (trained models + validation probs)
- `artifacts/features/` (precomputed tabular features)
- `*.joblib`, `*.pt`, `*.pth` (model binaries)
- `*.npy`, `*.npz` (large arrays)
