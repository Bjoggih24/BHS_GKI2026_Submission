from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
TRAIN_DIR = DATA_DIR / "train"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
SPLIT_DIR = ARTIFACTS_DIR / "split_seed42"
TABULAR_DIR = ARTIFACTS_DIR / "tabular"
CNN_DIR = ARTIFACTS_DIR / "cnn"
ENSEMBLE_PATH = ARTIFACTS_DIR / "ensemble_weight.json"
NORM_STATS_PATH = ARTIFACTS_DIR / "norm_stats.npz"
