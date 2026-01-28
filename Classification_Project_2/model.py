"""
Model for habitat classification.

This file contains the predict() function that will be called by the API.
"""

from pathlib import Path
import json
import os
import numpy as np
import torch

from config import ARTIFACTS_DIR, BASE_DIR
from feature_utils import extract_tabular_features_v2, extract_tabular_features_v3

_CACHE = {}


def _load_ensemble_config():
    override = os.environ.get("ENSEMBLE_CONFIG", "").strip()
    cfg_paths = []
    if override:
        override_path = Path(override)
        if override_path.is_absolute():
            cfg_paths.append(override_path)
        else:
            cfg_paths.append(BASE_DIR / override_path)
            cfg_paths.append(ARTIFACTS_DIR / override_path)
    cfg_paths += [
        BASE_DIR / "cnn" / "ensemble_submit.json",
        ARTIFACTS_DIR / "ensemble_submit.json",
        ARTIFACTS_DIR / "ensemble_topK.json",
        ARTIFACTS_DIR / "ensemble_top3.json",
    ]
    for path in cfg_paths:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return None


def _normalize_ensemble_config(cfg: dict) -> dict:
    if "models" in cfg:
        return cfg
    if "tabular_models" in cfg and "cnn_models" in cfg:
        models = []
        for entry in cfg.get("tabular_models", []):
            models.append(
                {
                    "type": "tabular",
                    "name": Path(entry["path"]).parent.name,
                    "model_path": entry["path"],
                    "weight": entry["weight"],
                }
            )
        for entry in cfg.get("cnn_models", []):
            models.append(
                {
                    "type": "cnn",
                    "name": Path(entry["path"]).parent.name,
                    "model_path": entry["path"],
                    "arch": entry.get("arch"),
                    "weight": entry["weight"],
                }
            )
        return {"models": models, "feature_version": cfg.get("feature_version")}
    raise ValueError("Unsupported ensemble config format. Expected 'models' list.")


def _lazy_load():
    if _CACHE:
        return

    import joblib
    import torch
    from torch import nn
    from cnn.model import make_cnn

    class SmallCNN(torch.nn.Module):
        def __init__(self, in_ch: int, num_classes: int):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Conv2d(in_ch, 32, kernel_size=3, padding=1, bias=False),
                torch.nn.BatchNorm2d(32),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(2),
                torch.nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
                torch.nn.BatchNorm2d(64),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(2),
                torch.nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
                torch.nn.BatchNorm2d(128),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(2),
                torch.nn.Conv2d(128, 192, kernel_size=3, padding=1, bias=False),
                torch.nn.BatchNorm2d(192),
                torch.nn.ReLU(inplace=True),
                torch.nn.AdaptiveAvgPool2d(1),
            )
            self.fc = torch.nn.Linear(192, num_classes)

        def forward(self, x):
            x = self.net(x).flatten(1)
            return self.fc(x)

    cfg = _load_ensemble_config()
    if not cfg:
        raise FileNotFoundError("No ensemble config found. Train models and run tuning.")
    cfg = _normalize_ensemble_config(cfg)
    feature_version = cfg.get("feature_version")
    if feature_version not in (None, "v2", "v3"):
        raise ValueError(
            f"Unsupported feature_version '{feature_version}'. Expected 'v2' or 'v3'."
        )
    if feature_version in (None, "v2"):
        _CACHE["feature_version"] = "v2"
    else:
        _CACHE["feature_version"] = "v3"

    tab_models = []
    tab_weights = []
    tab_meta = []
    cnn_models = []
    cnn_weights = []
    cnn_meta = []

    def _build_torch_mlp(params: dict, input_dim: int, num_classes: int) -> torch.nn.Module:
        hidden_dims = params.get("hidden_dims", [256, 128])
        dropout = float(params.get("dropout", 0.2))
        use_batch_norm = bool(params.get("batch_norm", False))
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            h = int(h)
            layers.append(nn.Linear(in_dim, h))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))
        return nn.Sequential(*layers)

    class TabTransformer(nn.Module):
        def __init__(
            self,
            num_features: int,
            num_classes: int,
            d_model: int,
            nhead: int,
            num_layers: int,
            dim_feedforward: int,
            dropout: float,
        ):
            super().__init__()
            self.feat_emb = nn.Linear(1, d_model)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            self.pos_emb = nn.Parameter(torch.zeros(1, num_features + 1, d_model))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.head = nn.Linear(d_model, num_classes)

        def forward(self, x):
            x = x.unsqueeze(-1)
            x = self.feat_emb(x)
            bsz = x.size(0)
            cls = self.cls_token.expand(bsz, -1, -1)
            x = torch.cat([cls, x], dim=1)
            x = x + self.pos_emb[:, : x.size(1), :]
            x = self.encoder(x)
            return self.head(x[:, 0])

    for entry in cfg["models"]:
        if entry["type"] == "tabular":
            model_path = ARTIFACTS_DIR / entry["model_path"]
            if model_path.suffix == ".pt" or entry.get("model_format") == "torch_state_dict":
                meta_path = model_path.with_name("meta.json")
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                model_kind = entry.get("model_type") or meta.get("model")
                params = meta.get("params_used") or meta.get("params") or {}
                standardize = bool(meta.get("standardize", False))
                mean = std = None
                if standardize:
                    npz = np.load(model_path.with_name("standardize.npz"))
                    mean = npz["mean"].astype(np.float32)
                    std = npz["std"].astype(np.float32)
                input_dim = int(len(mean)) if mean is not None else None
                if input_dim is None:
                    raise ValueError(f"Missing standardize stats for torch tabular model: {model_path}")
                num_classes = int(meta.get("num_classes", 71))
                if model_kind == "torch_mlp":
                    model = _build_torch_mlp(params, input_dim=input_dim, num_classes=num_classes)
                elif model_kind == "torch_transformer":
                    model = TabTransformer(
                        num_features=input_dim,
                        num_classes=num_classes,
                        d_model=int(params.get("d_model", 64)),
                        nhead=int(params.get("nhead", 4)),
                        num_layers=int(params.get("num_layers", 2)),
                        dim_feedforward=int(params.get("dim_feedforward", 128)),
                        dropout=float(params.get("dropout", 0.2)),
                    )
                else:
                    raise ValueError(f"Unsupported torch tabular model: {model_kind}")
                model.load_state_dict(torch.load(model_path, map_location="cpu"))
                model.eval()
                tab_models.append(model)
                tab_weights.append(entry["weight"])
                tab_meta.append({"type": model_kind, "mean": mean, "std": std})
            else:
                tab_models.append(joblib.load(model_path))
                tab_weights.append(entry["weight"])
                tab_meta.append({"type": "sklearn"})
        elif entry["type"] == "cnn":
            if not entry.get("arch"):
                raise ValueError("CNN entry missing 'arch' in ensemble config.")
            arch = entry["arch"]
            ckpt = ARTIFACTS_DIR / entry["model_path"]
            if arch == "small_cnn":
                norm_path = ARTIFACTS_DIR / entry["norm_path"]
                npz = np.load(norm_path)
                mean = npz["mean"].astype(np.float32)
                std = npz["std"].astype(np.float32)
                use_log1p = False
                if "use_log1p" in npz.files:
                    use_log1p = bool(npz["use_log1p"].item())
                model = SmallCNN(in_ch=15, num_classes=71)
                model.load_state_dict(torch.load(ckpt, map_location="cpu"))
                model.eval()
                cnn_models.append(model)
                cnn_weights.append(entry["weight"])
                cnn_meta.append(
                    {"arch": arch, "mean": mean, "std": std, "use_log1p": use_log1p}
                )
            else:
                model = make_cnn(arch, in_ch=15, num_classes=71)
                model.load_state_dict(torch.load(ckpt, map_location="cpu"))
                model.eval()
                cnn_models.append(model)
                cnn_weights.append(entry["weight"])
                cnn_meta.append({"arch": arch})
        else:
            raise ValueError(f"Unknown model type: {entry['type']}")

    if not tab_models and not cnn_models:
        raise FileNotFoundError("No models listed in ensemble config.")
    _CACHE["tabular_models"] = tab_models
    _CACHE["tabular_weights"] = tab_weights
    _CACHE["tabular_meta"] = tab_meta
    _CACHE["cnn_models"] = cnn_models
    _CACHE["cnn_weights"] = cnn_weights
    _CACHE["cnn_meta"] = cnn_meta
    _CACHE["weight_mode"] = "per-model"


def _predict_proba_best(model, feats: np.ndarray) -> np.ndarray:
    try:
        import xgboost as xgb
    except Exception:
        xgb = None

    if xgb is not None and isinstance(model, xgb.Booster):
        dmatrix = xgb.DMatrix(feats)
        if hasattr(model, "best_iteration") and model.best_iteration is not None:
            p = model.predict(dmatrix, iteration_range=(0, int(model.best_iteration) + 1))
        else:
            p = model.predict(dmatrix)
        if p.ndim == 1:
            return p
        return p[0]
    if hasattr(model, "best_iteration") and model.best_iteration is not None:
        return model.predict_proba(feats, iteration_range=(0, int(model.best_iteration) + 1))[0]
    if hasattr(model, "best_iteration_") and model.best_iteration_ is not None:
        return model.predict_proba(feats, num_iteration=int(model.best_iteration_))[0]
    return model.predict_proba(feats)[0]


def predict(patch: np.ndarray) -> int:
    """
    Predict habitat class for a single patch.
    """
    _lazy_load()

    probs_total = np.zeros(71, dtype=np.float32)

    # Tabular ensemble
    if _CACHE.get("feature_version") == "v3":
        feats = extract_tabular_features_v3(patch)[None, :]
    else:
        feats = extract_tabular_features_v2(patch)[None, :]
    for model, meta, w in zip(
        _CACHE["tabular_models"], _CACHE["tabular_meta"], _CACHE["tabular_weights"]
    ):
        if meta.get("type") in {"torch_mlp", "torch_transformer"}:
            x = feats.astype(np.float32, copy=False)
            mean = meta.get("mean")
            std = meta.get("std")
            if mean is not None and std is not None:
                x = (x - mean) / std
            with torch.no_grad():
                logits = model(torch.from_numpy(x))
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0].astype(np.float32)
            probs_total += probs * float(w)
        else:
            probs_total += _predict_proba_best(model, feats).astype(np.float32) * float(w)

    # CNN ensemble
    def _preprocess_small_cnn(p: np.ndarray, mean: np.ndarray, std: np.ndarray, use_log1p: bool) -> np.ndarray:
        x = p.astype(np.float32, copy=False)
        if use_log1p:
            x = np.log1p(np.maximum(x, 0.0))
        x = (x - mean[:, None, None]) / std[:, None, None]
        return x

    with torch.no_grad():
        for model, meta, w in zip(_CACHE["cnn_models"], _CACHE["cnn_meta"], _CACHE["cnn_weights"]):
            if meta.get("arch") == "small_cnn":
                x = _preprocess_small_cnn(patch, meta["mean"], meta["std"], meta["use_log1p"])
                x_t = torch.from_numpy(x).unsqueeze(0)
            else:
                raise RuntimeError(f"Unsupported arch in submission: {meta.get('arch')}")
            logits = model(x_t)
            probs_total += torch.softmax(logits, dim=1).cpu().numpy()[0].astype(np.float32) * float(w)

    return int(np.argmax(probs_total))


def baseline_model(patch: np.ndarray) -> int:
    raise RuntimeError("Baseline disabled. Train models and use predict().")
