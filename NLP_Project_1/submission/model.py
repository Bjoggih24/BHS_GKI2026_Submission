"""
Simple n-gram model loader for next-byte prediction.

Supports:
- Baseline heuristic model (no training)
- Legacy JSON counts (raw counts)
- Smoothed JSON model (format_version=2)
"""

import math
import json
from pathlib import Path


class Model:
    def __init__(self, submission_dir: Path):
        """
        Initialize the model.

        This baseline uses hardcoded byte frequencies typical of Icelandic text.
        For better results, train on the actual dataset!
        """
        # Try to load trained counts if available
        self.gru_model = None
        self.gru_config = None

        gru_weights = submission_dir / "gru_weights.npz"
        gru_config = submission_dir / "gru_config.json"
        if gru_weights.exists() and gru_config.exists():
            self._load_gru_model(gru_weights, gru_config)
        else:
            counts_file = submission_dir / "counts.json.gz"
            if counts_file.exists():
                self._load_trained_model(counts_file)
            else:
                self._init_baseline()

    def _init_baseline(self):
        """Initialize with a simple baseline (no training needed)."""
        # Basic ASCII/UTF-8 byte frequencies for text
        # Common bytes: space, lowercase letters, newlines
        self.logits_cache = {}

        # Default logits: slightly favor common text bytes
        self.default_logits = [0.0] * 256

        # Boost common ASCII characters
        # Space (32) is very common
        self.default_logits[32] = 3.0  # space
        self.default_logits[10] = 1.5  # newline

        # Lowercase letters (97-122) are common
        for i in range(97, 123):
            self.default_logits[i] = 2.0

        # Uppercase letters (65-90) less common
        for i in range(65, 91):
            self.default_logits[i] = 1.0

        # Digits (48-57)
        for i in range(48, 58):
            self.default_logits[i] = 0.5

        # Common punctuation
        self.default_logits[46] = 1.0  # period
        self.default_logits[44] = 1.0  # comma

        # Icelandic-specific: UTF-8 continuation bytes are common
        # UTF-8 continuation bytes: 128-191
        for i in range(128, 192):
            self.default_logits[i] = 1.0

        # UTF-8 2-byte start: 192-223
        for i in range(192, 224):
            self.default_logits[i] = 0.5

        self.trained = False
        print("Using baseline model (no training data)")

    def _load_trained_model(self, counts_file: Path):
        """Load trained n-gram model from JSON."""
        import gzip
        import json

        with gzip.open(counts_file, 'rt') as f:
            raw = json.load(f)

        if isinstance(raw, dict) and "meta" in raw:
            self._load_smoothed_model(raw)
        else:
            self._load_legacy_counts(raw)

    def _load_smoothed_model(self, raw: dict):
        meta = raw.get("meta", {})
        if meta.get("format_version") != 2:
            raise ValueError("Unsupported model format_version")

        quant_scale = int(meta.get("quant_scale", 1024))
        self.max_ctx_len = int(meta["n"]) - 1
        self.store_mode = meta.get("store", "delta")
        self.strategy = meta.get("strategy", "additive")

        q_unigram = raw.get("unigram_logp_q")
        if not q_unigram:
            raise ValueError("Missing unigram_logp_q in model")
        self.unigram_logits = tuple(q / quant_scale for q in q_unigram)

        tables = {}
        raw_tables = raw.get("tables", {})
        for L_str, entries in raw_tables.items():
            L = int(L_str)
            tbl = {}
            for ctx_key, deltas in entries:
                ctx_bytes = ctx_key.encode("latin1")
                tbl[ctx_bytes] = [(b, q / quant_scale) for b, q in deltas]
            tables[L] = tbl
        self.tables = tables

        self.cache = {}
        self.cache_max = 200_000

        self.trained = True
        print(f"Loaded smoothed JSON model (n={self.max_ctx_len + 1}, store={self.store_mode})")

    def _load_legacy_counts(self, raw_counts: dict):
        """Load legacy counts.json.gz format."""
        import json

        self.counts = {}
        for context_str, byte_counts in raw_counts.items():
            context = tuple(json.loads(context_str))
            self.counts[context] = {b: c for b, c in byte_counts}

        self.unigram = [1] * 256
        for byte_counts in self.counts.values():
            for byte, count in byte_counts.items():
                self.unigram[byte] += count

        self.unigram_logits = [math.log(c + 1) for c in self.unigram]

        self.trained = True
        print(f"Loaded legacy JSON counts with {len(self.counts)} contexts")

    def predict(self, contexts: list[list[int]]) -> list[list[float]]:
        """
        Predict next byte for each context.

        Args:
            contexts: List of byte sequences (each is list of ints 0-255)

        Returns:
            List of logit vectors, shape [batch_size, 256]
        """
        if self.gru_model is not None:
            return self._predict_gru(contexts)
        if not self.trained:
            return [self._predict_baseline(ctx) for ctx in contexts]
        if hasattr(self, "tables"):
            return [self._predict_smoothed(ctx) for ctx in contexts]
        return [self._predict_trained(ctx) for ctx in contexts]

    def _predict_baseline(self, context: list[int]) -> list[float]:
        """Simple baseline prediction."""
        # Use context to slightly adjust predictions
        logits = self.default_logits.copy()

        if len(context) > 0:
            last_byte = context[-1]

            # After space, boost lowercase letters
            if last_byte == 32:
                for i in range(97, 123):
                    logits[i] += 1.0

            # After newline, boost uppercase/space
            elif last_byte == 10:
                for i in range(65, 91):
                    logits[i] += 0.5
                logits[32] += 0.5

            # After period, boost space
            elif last_byte == 46:
                logits[32] += 2.0
                logits[10] += 1.0

        return logits

    def _predict_trained(self, context: list[int]) -> list[float]:
        """Prediction using trained n-gram counts."""
        # Try progressively shorter contexts
        for length in range(min(len(context), 6), -1, -1):
            if length == 0:
                break
            ctx_tuple = tuple(context[-length:])
            if ctx_tuple in self.counts:
                return self._counts_to_logits(self.counts[ctx_tuple])

        return self.unigram_logits

    def _predict_smoothed(self, context: list[int]) -> list[float]:
        """Prediction using smoothed backoff tables (format_version=2)."""
        tail = bytes(context[-self.max_ctx_len:]) if self.max_ctx_len > 0 else b""
        hit = self.cache.get(tail)
        if hit is not None:
            return hit

        logits = list(self.unigram_logits)
        ctx_len = len(context)

        if self.store_mode == "abs":
            for L in range(min(self.max_ctx_len, ctx_len), 0, -1):
                ctx_bytes = bytes(context[-L:])
                entry = self.tables.get(L, {}).get(ctx_bytes)
                if entry:
                    for b, lp in entry:
                        logits[b] = lp
                    break
            if len(self.cache) >= self.cache_max:
                self.cache.clear()
            self.cache[tail] = logits
            return logits

        for L in range(1, self.max_ctx_len + 1):
            if ctx_len < L:
                break
            ctx_bytes = bytes(context[-L:])
            entry = self.tables.get(L, {}).get(ctx_bytes)
            if entry:
                for b, delta in entry:
                    logits[b] += delta
        if len(self.cache) >= self.cache_max:
            self.cache.clear()
        self.cache[tail] = logits
        return logits

    def _load_gru_model(self, weights_path: Path, config_path: Path):
        import numpy as np
        import torch
        import torch.nn as nn

        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        from torch.nn.utils.rnn import pack_padded_sequence

        class TinyGRU(nn.Module):
            def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, layers: int):
                super().__init__()
                self.embed = nn.Embedding(vocab_size, embed_dim)
                self.gru = nn.GRU(embed_dim, hidden_dim, num_layers=layers, batch_first=True)
                self.fc = nn.Linear(hidden_dim, vocab_size)

            from torch.nn.utils.rnn import pack_padded_sequence

            def forward(self, x, lengths):
                emb = self.embed(x)

                # If all sequences are full length, skip packing entirely (fast path)
                if int(lengths.min()) == x.size(1) and int(lengths.max()) == x.size(1):
                    _, h_n = self.gru(emb)  # h_n: [layers, B, H]
                else:
                    packed = pack_padded_sequence(
                        emb, lengths.cpu(), batch_first=True, enforce_sorted=False
                    )
                    _, h_n = self.gru(packed)

                last = h_n[-1]            # [B, H] last layer
                return self.fc(last)      # [B, 256]




        model = TinyGRU(256, cfg["embed_dim"], cfg["hidden_dim"], cfg["layers"])

        weights = np.load(weights_path)
        state = {}
        for k in weights.files:
            if k.endswith("__scale"):
                continue
            q = weights[k].astype(np.float32)
            s_key = k + "__scale"
            if s_key not in weights.files:
                raise ValueError(f"Missing scale for {k}")
            scale = float(weights[s_key].astype(np.float32))
            w = q * scale
            state[k] = torch.tensor(w, dtype=torch.float32)
        model.load_state_dict(state)
        model.eval()

        self.gru_model = model
        self.gru_config = cfg
        self.max_ctx_len = int(cfg.get("max_ctx_len", 256))
        self.cache_len = int(cfg.get("cache_len", 128))
        self.cache_weight = float(cfg.get("cache_weight", 0.0))
        print(f"Loaded GRU model (embed={cfg['embed_dim']}, hidden={cfg['hidden_dim']})")

    def _predict_gru(self, contexts: list[list[int]]) -> list[list[float]]:
        import torch
        import math

        max_len = self.max_ctx_len
        device = next(self.gru_model.parameters()).device
        batch_size = len(contexts)
        lengths = [min(len(c), max_len) for c in contexts]
        seq_len = max(lengths) if lengths else 1
        if seq_len == 0:
            seq_len = 1

        x = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)
        for i, ctx in enumerate(contexts):
            tail = ctx[-max_len:] if max_len > 0 else []
            if not tail:
                continue
            x[i, : len(tail)] = torch.tensor(tail, dtype=torch.long, device=device)

        lengths_t = torch.tensor([max(1, l) for l in lengths], dtype=torch.long, device=device)
        with torch.no_grad():
            raw_logits = self.gru_model(x, lengths_t)
            gru_logp = torch.log_softmax(raw_logits, dim=1)

        if self.cache_weight <= 0.0 or self.cache_len <= 0:
            return gru_logp.cpu().tolist()

        # Build cache counts tensor [B, 256]
        cache_len = self.cache_len
        w_cache = min(max(self.cache_weight, 0.0), 0.999)
        if w_cache >= 1.0:
            w_cache = 0.999
        cache_lengths = [min(len(c), cache_len) for c in contexts]
        max_cache_len = max(cache_lengths) if cache_lengths else 1
        if max_cache_len == 0:
            return gru_logp.cpu().tolist()

        tail_tokens = torch.zeros((batch_size, max_cache_len), dtype=torch.long, device=device)
        mask = torch.zeros((batch_size, max_cache_len), dtype=torch.float32, device=device)
        for i, ctx in enumerate(contexts):
            if cache_lengths[i] == 0:
                continue
            tail = ctx[-cache_lengths[i]:]
            tail_tokens[i, : len(tail)] = torch.tensor(tail, dtype=torch.long, device=device)
            mask[i, : len(tail)] = 1.0

        counts = torch.zeros((batch_size, 256), dtype=torch.float32, device=device)
        counts.scatter_add_(1, tail_tokens, mask)

        alpha = 0.01
        denom = counts.sum(dim=1, keepdim=True) + alpha * 256.0
        cache_logp = torch.log((counts + alpha) / denom)

        log_w_cache = math.log(w_cache)
        log_w_gru = math.log(1.0 - w_cache)
        mixed = torch.logaddexp(gru_logp + log_w_gru, cache_logp + log_w_cache)
        return mixed.cpu().tolist()

    def _counts_to_logits(self, byte_counts: dict) -> list[float]:
        """Convert counts to logits."""
        logits = [0.0] * 256
        for byte, count in byte_counts.items():
            logits[byte] = math.log(count + 1)
        return logits