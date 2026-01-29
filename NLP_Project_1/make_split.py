import json, random
from pathlib import Path
from datasets import load_from_disk

DATA = Path("data/igc_full")
OUT = Path("data/splits")
OUT.mkdir(parents=True, exist_ok=True)

SEED = 1337
VAL_FRAC = 0.02   # 2%
TEST_FRAC = 0.01  # 1% (optional)

if not DATA.exists():
    raise FileNotFoundError("Missing data/igc_full. Run create_dataset.py first.")

ds = load_from_disk(str(DATA))
n = len(ds)

idx = list(range(n))
random.Random(SEED).shuffle(idx)

n_test = int(n * TEST_FRAC)
n_val = int(n * VAL_FRAC)

test_idx = idx[:n_test]
val_idx  = idx[n_test:n_test+n_val]
train_idx= idx[n_test+n_val:]

(OUT/"train_idx.json").write_text(json.dumps(train_idx))
(OUT/"val_idx.json").write_text(json.dumps(val_idx))
(OUT/"test_idx.json").write_text(json.dumps(test_idx))

print(f"Total docs: {n}")
print(f"Train: {len(train_idx)}  Val: {len(val_idx)}  Test: {len(test_idx)}")
print(f"Wrote splits to: {OUT}/")
