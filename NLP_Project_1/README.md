
---

# Project 1 - Golden Plate at Thingvellir (Byte-level LM)

## Goal
Train a tiny **byte-level (0-255)** next-byte prediction model under strict constraints:
- **CPU-only inference** in evaluation
- **Time and memory constraints**
- **submission.zip <= 1 MB** (and uncompressed <= 50 MB)

Metric: **bits per byte (BPB)**. Lower is better.

## Final Results

**Validation Score (Leaderboard):** 1.5478 bits-per-byte (BPB)

**Test Score (Leaderboard):** 1.51 bits-per-byte (BPB)

---

## Approach Summary
My best-performing approach is a **tiny GRU language model** with:
- `Embedding(256 -> embed_dim)`
- `GRU(embed_dim -> hidden_dim, layers = 1)`
- `Linear(hidden_dim -> 256 logits)`

To fit under the **1 MB submission limit**, weights are:
- **quantized to int8** per tensor
- stored as `int8 weights + fp16 scale` in a compressed `.npz`

The submission runtime loads weights, reconstructs float weights, and performs fast batched inference.

---

## Key Files

### Training + Export
- `scripts/train_tiny_gru.py` - trains the GRU model and exports quantized weights/config

### Submission (what the evaluator uses)
- `submission/model.py` - implements the competition `Model` interface; loads quantized weights and predicts next-byte logits
- `submission/gru_config.json` - model hyperparameters (embedding dim, hidden dim, layers, etc.)

### Packaging + Validation
- `create_submission.py` - builds `submission.zip` from the `submission/` folder and prints size checks
- `check_submission.py` - validates zip structure, imports `model.py`, runs batch-shape and timing tests
- `score_local.py` - local BPB evaluation

### Data Preparation
- `create_dataset.py` - downloads and prepares dataset locally in `data/igc_full/`
- `make_split.py` - creates train, val, and test split index files in `data/splits/`

### Configuration
- `requirements.txt` - Python dependencies

### Generated Outputs (after running the steps below)
- `submission/gru_weights.npz` - quantized weights
- `submission.zip` - submission archive

## Setup

### 1) Create environment
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2) Prepare data locally
```bash
python create_dataset.py
python make_split.py
```

This creates:
- `data/igc_full/` - downloaded and processed text data
- `data/splits/train_idx.json` - training split indices
- `data/splits/val_idx.json` - validation split indices
- `data/splits/test_idx.json` - test split indices (optional)

### 3) Train the model

Example: 1-layer GRU (competitive baseline)
```bash
python scripts/train_tiny_gru.py \
  --data data/igc_full \
  --max-docs 150000 \
  --epochs 20 \
  --samples-per-epoch 1000000 \
  --seq-len 256 \
  --batch-size 256 \
  --embed-dim 256 \
  --hidden-dim 544 \
  --layers 1 \
  --lr 6e-4 \
  --cache-weight 0.00 \
  --device cuda \
  --amp
```

**Notes:**
- `hidden_dim=544` is near the size limit (weights barely fit under 1 MB after quantization and compression)
- Remove `--amp` on CPU-only systems and use `--device cpu`
- Weights are automatically quantized and exported to `submission/`
- The final submission model was trained for roughly **40 to 50 epochs**

### 4) Validate submission

Check submission structure and test inference:
```bash
python check_submission.py
```

### 5) Local evaluation

Evaluate BPB on validation split:
```bash
python score_local.py
```

**Note:** `score_local.py` samples random next-byte positions from the val split; it does **not** iterate sequentially over a fixed held-out document set. Treat it as a quick sanity check rather than a strict deterministic validation pass.

### 6) Build final submission zip

```bash
python create_submission.py
```

This creates `submission.zip` (should be <= 1 MB).


