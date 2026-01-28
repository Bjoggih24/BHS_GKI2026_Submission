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
- `api.py` -FastAPI `/predict` endpoint.
- `model.py` -inference entry point; loads ensemble config and models.
- `pyproject.toml` -project configuration and dependencies.

Feature Engineering + Utilities:
- `feature_utils.py` -tabular feature engineering (v1/v2/v3).
- `utils.py` -patch encoding/decoding and data helpers.
- `config.py` -paths for data/artifacts/splits.

CNN:
- `cnn/dataset.py` -CNN dataset loading utilities.
- `cnn/model.py` -CNN model implementations.
- `cnn/ensemble_submit.json` -final ensemble configuration for API inference.

Training + Evaluation Scripts:
- `scripts/precompute_tabular_features.py` -build `X_tabular_v*.npy`.
- `scripts/train_tabular.py` -train ET/LGBM/XGB/MLP/Transformer tabular models.
- `scripts/train_cnn.py` -train CNN models (small_cnn and timm backbones).
- `scripts/simple_ensemble_weights.py` -quick weighting tests.

Artifacts (generated during training):
- `artifacts/features/` -precomputed tabular features (`.npy` files).
- `artifacts/experiments/` -trained model checkpoints and validation probabilities.
- `artifacts/split_seed42/` -train/val index splits (required).
- Other `artifacts/ensemble_*.json` files -alternative ensemble configurations.

## Final Results

**Validation Score (Leaderboard):** Weighted F1: 0.3899
**Test Scorer (Leaderbaord):** Weighted F1: 0.39

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

2. **Training data** in `data/train/`:
   - `patches.npy` -raw satellite imagery patches
   - `train.csv` -labels (column: `vistgerd_idx`)

3. **Submission configs** in `submission_configs/`:
   - Only the configs used by the final ensemble

## Reproducible Pipeline

### 1) Prepare data + splits
```bash
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
Quick weighting (validation probs):
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

