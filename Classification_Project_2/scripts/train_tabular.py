#!/usr/bin/env python3
"""
Train tabular models from precomputed features.
"""

import sys
import argparse
import json
import inspect
import os
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import SPLIT_DIR
from scripts.eval_utils import load_split_indices, weighted_f1, macro_f1, report_per_class_f1


def _jsonable(value):
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def build_model(model_name: str, params: dict):
    if model_name == "extratrees":
        return ExtraTreesClassifier(**params)
    if model_name == "randomforest":
        return RandomForestClassifier(**params)
    if model_name == "hgb":
        return HistGradientBoostingClassifier(**params)
    if model_name == "xgb":
        import xgboost as xgb

        return xgb.XGBClassifier(**params)
    if model_name == "lgbm":
        import lightgbm as lgb

        return lgb.LGBMClassifier(**params)
    if model_name == "mlp":
        from sklearn.neural_network import MLPClassifier

        return MLPClassifier(**params)
    raise ValueError(f"Unknown model: {model_name}")


def predict_proba_best(model, X):
    try:
        import xgboost as xgb
    except Exception:
        xgb = None

    if xgb is not None and isinstance(model, xgb.Booster):
        dmatrix = xgb.DMatrix(X)
        if hasattr(model, "best_iteration") and model.best_iteration is not None:
            p = model.predict(dmatrix, iteration_range=(0, int(model.best_iteration) + 1))
        else:
            p = model.predict(dmatrix)
        p = np.asarray(p)
        if p.ndim == 1:
            p = p.reshape(-1, 1)
        return p

    if hasattr(model, "best_iteration") and model.best_iteration is not None:
        return model.predict_proba(X, iteration_range=(0, int(model.best_iteration) + 1))
    if hasattr(model, "best_iteration_") and model.best_iteration_ is not None:
        return model.predict_proba(X, num_iteration=int(model.best_iteration_))
    return model.predict_proba(X)


def train_torch_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    num_classes: int,
    params: dict,
    class_weight: np.ndarray | None,
    seed: int,
):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    torch.set_num_threads(4)

    hidden_dims = params.get("hidden_dims", [256, 128])
    dropout = float(params.get("dropout", 0.2))
    lr = float(params.get("lr", 1e-3))
    weight_decay = float(params.get("weight_decay", 1e-4))
    batch_size = int(params.get("batch_size", 128))
    epochs = int(params.get("epochs", 200))
    patience = int(params.get("patience", 20))
    standardize = bool(params.get("standardize", True))
    label_smoothing = float(params.get("label_smoothing", 0.0))
    max_grad_norm = params.get("max_grad_norm", None)
    use_batch_norm = bool(params.get("batch_norm", False))

    if standardize:
        mean = X_train.mean(axis=0, dtype=np.float64)
        std = X_train.std(axis=0, dtype=np.float64)
        std = np.maximum(std, 1e-6)
        X_train = (X_train - mean) / std
        X_val = (X_val - mean) / std
    else:
        mean = None
        std = None

    X_train_t = torch.from_numpy(X_train.astype(np.float32, copy=False))
    y_train_t = torch.from_numpy(y_train.astype(np.int64, copy=False))
    X_val_t = torch.from_numpy(X_val.astype(np.float32, copy=False))

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    layers = []
    in_dim = X_train.shape[1]
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
    model = nn.Sequential(*layers)

    loss_kwargs = {}
    try:
        if label_smoothing > 0:
            loss_kwargs["label_smoothing"] = label_smoothing
    except Exception:
        loss_kwargs = {}
    if class_weight is not None:
        loss_kwargs["weight"] = torch.from_numpy(class_weight.astype(np.float32))
    criterion = nn.CrossEntropyLoss(**loss_kwargs)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_wf1 = -1.0
    best_epoch = -1
    best_state = None
    patience_left = patience

    def eval_val():
        model.eval()
        with torch.no_grad():
            logits = model(X_val_t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        pred = np.argmax(probs, axis=1)
        wf1 = weighted_f1(y_val, pred)
        mf1 = macro_f1(y_val, pred)
        return probs, wf1, mf1

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            if max_grad_norm:
                nn.utils.clip_grad_norm_(model.parameters(), float(max_grad_norm))
            optimizer.step()

        _, wf1, _ = eval_val()
        if wf1 > best_wf1:
            best_wf1 = wf1
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    probs, wf1, mf1 = eval_val()
    return model, probs, wf1, mf1, best_epoch, mean, std


def train_torch_transformer(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    num_classes: int,
    params: dict,
    class_weight: np.ndarray | None,
    seed: int,
):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    torch.set_num_threads(4)

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

    d_model = int(params.get("d_model", 64))
    nhead = int(params.get("nhead", 4))
    num_layers = int(params.get("num_layers", 2))
    dim_feedforward = int(params.get("dim_feedforward", 128))
    dropout = float(params.get("dropout", 0.2))
    lr = float(params.get("lr", 1e-3))
    weight_decay = float(params.get("weight_decay", 1e-4))
    batch_size = int(params.get("batch_size", 128))
    epochs = int(params.get("epochs", 200))
    patience = int(params.get("patience", 20))
    standardize = bool(params.get("standardize", True))
    label_smoothing = float(params.get("label_smoothing", 0.0))
    max_grad_norm = params.get("max_grad_norm", None)
    log_every = int(params.get("log_every", 1))

    if standardize:
        mean = X_train.mean(axis=0, dtype=np.float64)
        std = X_train.std(axis=0, dtype=np.float64)
        std = np.maximum(std, 1e-6)
        X_train = (X_train - mean) / std
        X_val = (X_val - mean) / std
    else:
        mean = None
        std = None

    X_train_t = torch.from_numpy(X_train.astype(np.float32, copy=False))
    y_train_t = torch.from_numpy(y_train.astype(np.int64, copy=False))
    X_val_t = torch.from_numpy(X_val.astype(np.float32, copy=False))

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    model = TabTransformer(
        num_features=X_train.shape[1],
        num_classes=num_classes,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
    )

    loss_kwargs = {}
    try:
        if label_smoothing > 0:
            loss_kwargs["label_smoothing"] = label_smoothing
    except Exception:
        loss_kwargs = {}
    if class_weight is not None:
        loss_kwargs["weight"] = torch.from_numpy(class_weight.astype(np.float32))
    criterion = nn.CrossEntropyLoss(**loss_kwargs)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_wf1 = -1.0
    best_epoch = -1
    best_state = None
    patience_left = patience

    def eval_val():
        model.eval()
        with torch.no_grad():
            logits = model(X_val_t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        pred = np.argmax(probs, axis=1)
        wf1 = weighted_f1(y_val, pred)
        mf1 = macro_f1(y_val, pred)
        return probs, wf1, mf1

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        for xb, yb in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            if max_grad_norm:
                nn.utils.clip_grad_norm_(model.parameters(), float(max_grad_norm))
            optimizer.step()
            total_loss += float(loss.item()) * xb.size(0)
            total_count += xb.size(0)

        _, wf1, mf1 = eval_val()
        if log_every > 0 and (epoch == 1 or epoch % log_every == 0):
            avg_loss = total_loss / max(total_count, 1)
            print(f"[transformer] epoch={epoch} loss={avg_loss:.4f} wf1={wf1:.4f} mf1={mf1:.4f}")
        if wf1 > best_wf1:
            best_wf1 = wf1
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                if log_every > 0:
                    print(f"[transformer] early_stop at epoch={epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    probs, wf1, mf1 = eval_val()
    return model, probs, wf1, mf1, best_epoch, mean, std


def main():
    parser = argparse.ArgumentParser(description="Train tabular models")
    parser.add_argument(
        "--model",
        choices=["extratrees", "randomforest", "hgb", "xgb", "lgbm", "mlp", "torch_mlp", "torch_transformer"],
        required=True,
    )
    parser.add_argument("--params_json", type=Path, required=True)
    parser.add_argument("--split_dir", type=Path, default=SPLIT_DIR)
    parser.add_argument("--features_path", type=Path, required=True)
    parser.add_argument("--labels_path", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_class_weights", type=int, default=1)
    parser.add_argument("--weight_power", type=float, default=0.25)
    args = parser.parse_args()

    with open(args.params_json, "r", encoding="utf-8") as f:
        params_raw = json.load(f)

    params_base = params_raw.copy()
    params_base["random_state"] = args.seed if "random_state" not in params_base else params_base["random_state"]
    params_used = params_base.copy()

    X_all = np.load(args.features_path)
    y_all = np.load(args.labels_path)
    num_classes = int(np.max(y_all)) + 1

    train_idx, val_idx = load_split_indices(args.split_dir)
    X_train = X_all[train_idx]
    y_train = y_all[train_idx]
    X_val = X_all[val_idx]
    y_val = y_all[val_idx]

    if args.model == "xgb":
        params_base.setdefault("eval_metric", "mlogloss")
        params_used = params_base.copy()

    model = None
    if args.model not in {"torch_mlp", "torch_transformer"}:
        model = build_model(args.model, params_base)
    sample_weight = None
    val_weight = None
    class_weight = None
    if args.use_class_weights:
        counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
        class_weight = 1.0 / np.power(np.maximum(counts, 1.0), args.weight_power)
        sample_weight = class_weight[y_train]
        sample_weight /= np.mean(sample_weight)
        val_weight = class_weight[y_val]
        val_weight /= np.mean(val_weight)

    fit_kwargs = {}
    use_xgb_train = False
    if args.model == "xgb":
        fit_kwargs = {"eval_set": [(X_val, y_val)], "verbose": False}
        es_rounds = int(params_base.get("early_stopping_rounds", 200))
        fit_sig = inspect.signature(model.fit).parameters
        if val_weight is not None and "sample_weight_eval_set" in fit_sig:
            fit_kwargs["sample_weight_eval_set"] = [val_weight]
        if "callbacks" in fit_sig:
            import xgboost as xgb

            fit_kwargs["callbacks"] = [xgb.callback.EarlyStopping(rounds=es_rounds, save_best=True)]
        elif "early_stopping_rounds" in fit_sig:
            fit_kwargs["early_stopping_rounds"] = es_rounds
        else:
            use_xgb_train = True
    if args.model == "lgbm":
        import lightgbm as lgb

        fit_kwargs = {
            "eval_set": [(X_val, y_val)],
            "eval_metric": "multi_logloss",
            "callbacks": [
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(period=50),
            ],
        }
        fit_sig = inspect.signature(model.fit).parameters
        if val_weight is not None and "eval_sample_weight" in fit_sig:
            fit_kwargs["eval_sample_weight"] = [val_weight]

    if args.model == "torch_mlp":
        model, val_probs, wf1, mf1, best_iter, mean, std = train_torch_mlp(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            num_classes=num_classes,
            params=params_base,
            class_weight=class_weight if args.use_class_weights else None,
            seed=args.seed,
        )
        best_score = None
        params_used = params_base.copy()
    elif args.model == "torch_transformer":
        model, val_probs, wf1, mf1, best_iter, mean, std = train_torch_transformer(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            num_classes=num_classes,
            params=params_base,
            class_weight=class_weight if args.use_class_weights else None,
            seed=args.seed,
        )
        best_score = None
        params_used = params_base.copy()
    elif args.model == "xgb" and use_xgb_train:
        import xgboost as xgb

        params_used = params_base.copy()
        num_boost_round = int(params_used.pop("n_estimators", 2000))
        es_rounds = int(params_used.pop("early_stopping_rounds", 200))
        n_jobs = params_used.pop("n_jobs", None)
        if n_jobs is None:
            nthread = os.cpu_count() or 1
        else:
            n_jobs = int(n_jobs)
            nthread = (os.cpu_count() or 1) if n_jobs < 0 else max(1, n_jobs)
        params_used.pop("random_state", None)
        params_used.setdefault("seed", args.seed)
        params_used.setdefault("nthread", nthread)
        dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weight, nthread=nthread)
        dval = xgb.DMatrix(X_val, label=y_val, weight=val_weight, nthread=nthread)
        print(
            f"[xgb.train] nthread={nthread} num_boost_round={num_boost_round} early_stopping_rounds={es_rounds}"
        )
        model = xgb.train(
            params_used,
            dtrain,
            num_boost_round=num_boost_round,
            evals=[(dtrain, "train"), (dval, "val")],
            early_stopping_rounds=es_rounds,
            verbose_eval=50,
        )
    else:
        if args.model == "extratrees":
            print("[extratrees] fitting...")
        model.fit(X_train, y_train, sample_weight=sample_weight, **fit_kwargs)
        if args.model == "extratrees":
            print("[extratrees] done.")

    if args.model not in {"torch_mlp", "torch_transformer"}:
        val_probs = predict_proba_best(model, X_val)
        val_pred = np.argmax(val_probs, axis=1)
        wf1 = weighted_f1(y_val, val_pred)
        mf1 = macro_f1(y_val, val_pred)
        worst = report_per_class_f1(y_val, val_pred, top_k=10)
        best_iter = getattr(model, "best_iteration", None)
        if best_iter is None:
            best_iter = getattr(model, "best_iteration_", None)
        best_score = getattr(model, "best_score", None)
        if best_score is None:
            best_score = getattr(model, "best_score_", None)
    else:
        val_pred = np.argmax(val_probs, axis=1)
        worst = report_per_class_f1(y_val, val_pred, top_k=10)
        best_score = None

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.model in {"torch_mlp", "torch_transformer"}:
        import torch

        torch.save(model.state_dict(), args.out_dir / "model.pt")
    else:
        joblib.dump(model, args.out_dir / "model.joblib")
    np.save(args.out_dir / "val_probs.npy", val_probs.astype(np.float32))

    feat_path = str(args.features_path)
    if "v3" in feat_path:
        feat_ver = "v3"
    elif "v2" in feat_path:
        feat_ver = "v2"
    else:
        feat_ver = "v1"
    meta = {
        "model": args.model,
        "params": params_raw,
        "params_used": params_used,
        "feature_version": feat_ver,
        "features_path": str(args.features_path),
        "labels_path": str(args.labels_path),
        "weighted_f1": wf1,
        "macro_f1": mf1,
        "worst_classes": worst,
        "best_iteration": None if best_iter is None else int(best_iter),
        "best_score": None if best_score is None else _jsonable(best_score),
    }
    if args.model in {"torch_mlp", "torch_transformer"}:
        meta["model_path"] = str(args.out_dir / "model.pt")
        meta["model_format"] = "torch_state_dict"
        meta["standardize"] = params_base.get("standardize", True)
        if mean is not None and std is not None:
            np.savez(args.out_dir / "standardize.npz", mean=mean.astype(np.float32), std=std.astype(np.float32))
    with open(args.out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    if args.model in {"torch_mlp", "torch_transformer"}:
        print(f"Saved {args.out_dir / 'model.pt'}")
    else:
        print(f"Saved {args.out_dir / 'model.joblib'}")
    print(f"Weighted F1: {wf1:.4f} | Macro F1: {mf1:.4f}")


if __name__ == "__main__":
    main()
