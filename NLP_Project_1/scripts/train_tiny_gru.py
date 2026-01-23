#!/usr/bin/env python3
"""
Train a tiny GRU byte-level language model and export weights to submission/.
"""

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader


def quantize_int8(arr: np.ndarray):
    arr32 = arr.astype(np.float32)
    mx = float(np.max(np.abs(arr32)))
    if mx < 1e-8:
        scale = np.float16(1.0)
        q = np.zeros_like(arr32, dtype=np.int8)
        return q, scale
    scale = np.float16(mx / 127.0)
    q = np.clip(np.round(arr32 / float(scale)), -127, 127).astype(np.int8)
    return q, scale


class RandomChunkDataset(IterableDataset):
    def __init__(self, docs: list[bytes], seq_len: int, samples_per_epoch: int, seed: int):
        super().__init__()
        self.docs = docs
        self.seq_len = seq_len
        self.samples_per_epoch = samples_per_epoch
        self.seed = seed

    def __iter__(self):
        rng = random.Random(self.seed)
        produced = 0
        while produced < self.samples_per_epoch:
            doc = self.docs[rng.randrange(len(self.docs))]
            if len(doc) <= self.seq_len + 1:
                continue
            start = rng.randrange(0, len(doc) - self.seq_len - 1)
            chunk = doc[start : start + self.seq_len + 1]
            x = np.frombuffer(chunk[:-1], dtype=np.uint8).astype(np.int64)
            y = np.frombuffer(chunk[1:], dtype=np.uint8).astype(np.int64)
            yield torch.from_numpy(x), torch.from_numpy(y)
            produced += 1


class TinyGRU(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, layers: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers=layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        emb = self.embed(x)
        out, _ = self.gru(emb)
        logits = self.fc(out)
        return logits


def load_docs(data_path: Path, max_docs: int | None):
    from datasets import load_from_disk

    dataset = load_from_disk(str(data_path))
    num_docs = min(len(dataset), max_docs) if max_docs else len(dataset)
    docs = []
    for i in range(num_docs):
        item = dataset[i]
        if "text" in item:
            docs.append(item["text"].encode("utf-8"))
    return docs


def export_weights(
    model: nn.Module,
    out_dir: Path,
    embed_dim: int,
    hidden_dim: int,
    layers: int,
    max_ctx_len: int,
    cache_len: int,
    cache_weight: float,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    state = model.state_dict()
    arrays = {}
    for k, v in state.items():
        q, s = quantize_int8(v.detach().cpu().numpy())
        arrays[k] = q
        arrays[k + "__scale"] = np.array(s, dtype=np.float16)
    np.savez_compressed(out_dir / "gru_weights.npz", **arrays)



    cfg = {
        "embed_dim": embed_dim,
        "hidden_dim": hidden_dim,
        "layers": layers,
        "max_ctx_len": max_ctx_len,
        "cache_len": cache_len,
        "cache_weight": cache_weight,
    }
    with open(out_dir / "gru_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Train tiny GRU byte LM")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--max-docs", type=int, default=20000)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--samples-per-epoch", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--max-ctx-len", type=int, default=256)
    parser.add_argument("--cache-len", type=int, default=128)
    parser.add_argument("--cache-weight", type=float, default=0.15)
    parser.add_argument("--out-dir", type=Path, default=Path("submission"))
    parser.add_argument("--ckpt-path", type=Path, default=Path("checkpoints/gru_ckpt.pt"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"])
    parser.add_argument("--amp", action="store_true", help="Use mixed precision (CUDA only)")
    args = parser.parse_args()

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    use_amp = bool(args.amp) and (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(f"[device] {device} | amp={use_amp}")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    docs = load_docs(args.data, args.max_docs)
    model = TinyGRU(256, args.embed_dim, args.hidden_dim, args.layers).to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    start_epoch = 0
    if args.resume and args.ckpt_path.exists():
        ckpt = torch.load(args.ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        # after opt.load_state_dict(ckpt["opt"])
        for pg in opt.param_groups:
            pg["lr"] = args.lr
        print(f"[resume] set optimizer lr={args.lr}")

        start_epoch = int(ckpt.get("epoch", 0))
        print(f"[resume] loaded checkpoint at epoch {start_epoch}")
        for state in opt.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)
        model.train()

    def save_ckpt(epoch_idx: int):
        args.ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        model_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        torch.save(
            {"epoch": epoch_idx, "model": model_state, "opt": opt.state_dict()},
            args.ckpt_path,
        )

    last_completed = start_epoch
    try:
        for epoch in range(start_epoch, args.epochs):
            dataset = RandomChunkDataset(docs, args.seq_len, args.samples_per_epoch, args.seed + epoch)
            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                drop_last=True,
                pin_memory=(device.type == "cuda"),
                num_workers=0,
            )
            total = 0.0
            count = 0
            for x, y in loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    logits = model(x)
                    loss = loss_fn(logits.reshape(-1, 256), y.reshape(-1))

                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                total += float(loss.item())
                count += 1
            avg_loss = total / max(count, 1)
            bpb = avg_loss / math.log(2.0)
            print(f"Epoch {epoch+1}/{args.epochs} loss={avg_loss:.4f} | bpb~{bpb:.4f}")
            last_completed = epoch + 1
            save_ckpt(last_completed)
    except KeyboardInterrupt:
        print("Interrupted, exporting current weights...")
        save_ckpt(last_completed)
        export_weights(
            model,
            args.out_dir,
            args.embed_dim,
            args.hidden_dim,
            args.layers,
            args.max_ctx_len,
            args.cache_len,
            args.cache_weight,
        )
        print("Saved GRU weights to submission/")
        return

    export_weights(
        model,
        args.out_dir,
        args.embed_dim,
        args.hidden_dim,
        args.layers,
        args.max_ctx_len,
        args.cache_len,
        args.cache_weight,
    )
    print("Saved GRU weights to submission/")


if __name__ == "__main__":
    main()
