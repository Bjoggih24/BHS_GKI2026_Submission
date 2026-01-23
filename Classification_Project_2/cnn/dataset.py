import numpy as np
import torch
from torch.utils.data import Dataset

from feature_utils import preprocess_cnn


class HabitatDataset(Dataset):
    def __init__(self, patches: np.ndarray, labels: np.ndarray, mean: np.ndarray, std: np.ndarray, augment: bool):
        self.patches = patches
        self.labels = labels
        self.mean = mean
        self.std = std
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def _augment(self, x: np.ndarray) -> np.ndarray:
        if np.random.rand() < 0.5:
            x = np.flip(x, axis=1)
        if np.random.rand() < 0.5:
            x = np.flip(x, axis=2)
        k = np.random.randint(0, 4)
        if k:
            x = np.rot90(x, k, axes=(1, 2))
        if np.random.rand() < 0.3:
            noise = np.random.normal(0, 0.01, size=x.shape).astype(np.float32)
            x = x + noise
        if np.random.rand() < 0.3:
            gains = np.random.uniform(0.9, 1.1, size=(x.shape[0], 1, 1)).astype(np.float32)
            x = x * gains
        return x

    def __getitem__(self, idx):
        patch = self.patches[idx]
        x = preprocess_cnn(patch, self.mean, self.std)
        if self.augment:
            x = self._augment(x)
        x = torch.from_numpy(np.ascontiguousarray(x))
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y
