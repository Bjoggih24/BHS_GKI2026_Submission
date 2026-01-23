import numpy as np


def aspect_to_sin_cos(aspect_deg: np.ndarray):
    aspect_rad = np.deg2rad(aspect_deg)
    return np.sin(aspect_rad), np.cos(aspect_rad)


def normalize_patch(patch: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (patch - mean[:, None, None]) / std[:, None, None]


def preprocess_cnn(patch: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    aspect_raw = patch[14]
    sin_aspect, cos_aspect = aspect_to_sin_cos(aspect_raw)
    patch = normalize_patch(patch, mean, std)
    elev = patch[12]
    slope = patch[13]
    bands = patch[:12]
    stacked = np.concatenate(
        [bands, elev[None], slope[None], sin_aspect[None], cos_aspect[None]],
        axis=0,
    )
    return stacked.astype(np.float32, copy=False)


def pooled_grid_features(patch: np.ndarray) -> np.ndarray:
    # patch: (C,35,35) -> (C,7,7) by 5x5 mean pooling
    c, h, w = patch.shape
    if h != 35 or w != 35:
        raise ValueError("Expected patch shape (C,35,35)")
    pooled = patch.reshape(c, 7, 5, 7, 5).mean(axis=(2, 4))
    return pooled.reshape(c * 49)


def extract_tabular_features(patch: np.ndarray) -> np.ndarray:
    means = patch.mean(axis=(1, 2))
    stds = patch.std(axis=(1, 2))
    grid = pooled_grid_features(patch)

    elev = patch[12]
    slope = patch[13]
    aspect = patch[14]
    sin_aspect, cos_aspect = aspect_to_sin_cos(aspect)

    terrain = np.array(
        [
            elev.mean(),
            elev.std(),
            slope.mean(),
            slope.std(),
            sin_aspect.mean(),
            cos_aspect.mean(),
        ],
        dtype=np.float32,
    )

    features = np.concatenate([means, stds, grid, terrain]).astype(np.float32)
    return features


def _mean_abs_grad(x: np.ndarray) -> np.ndarray:
    gx = np.abs(x[:, :, 1:] - x[:, :, :-1])
    gy = np.abs(x[:, 1:, :] - x[:, :-1, :])
    return (gx.mean(axis=(1, 2)) + gy.mean(axis=(1, 2))).astype(np.float32)


def extract_tabular_features_v2(patch: np.ndarray) -> np.ndarray:
    # Per-channel stats
    means = patch.mean(axis=(1, 2))
    stds = patch.std(axis=(1, 2))
    qs = np.quantile(patch, [0.1, 0.5, 0.9], axis=(1, 2))  # (3, C)
    mins = patch.min(axis=(1, 2))
    maxs = patch.max(axis=(1, 2))

    # Aspect handling
    aspect = patch[14]
    sin_aspect, cos_aspect = aspect_to_sin_cos(aspect)
    aspect_feats = np.array([sin_aspect.mean(), cos_aspect.mean()], dtype=np.float32)

    # Spectral indices (S2 band mapping assumptions)
    eps = 1e-6
    red = patch[3]
    green = patch[2]
    nir = patch[7]
    swir1 = patch[10]
    swir2 = patch[11]

    ndvi = (nir - red) / (nir + red + eps)
    ndwi = (green - nir) / (green + nir + eps)
    ndmi = (nir - swir1) / (nir + swir1 + eps)
    nbr = (nir - swir2) / (nir + swir2 + eps)
    idx_stack = np.stack([ndvi, ndwi, ndmi, nbr], axis=0)
    idx_means = idx_stack.mean(axis=(1, 2))
    idx_stds = idx_stack.std(axis=(1, 2))

    # Cheap texture on key bands + terrain
    grad_channels = np.array([3, 7, 10, 11, 12, 13], dtype=int)
    grad_feats = _mean_abs_grad(patch[grad_channels])

    # Grid pooling (v1 feature)
    grid = pooled_grid_features(patch)

    features = np.concatenate(
        [
            means,
            stds,
            qs.reshape(-1),
            mins,
            maxs,
            aspect_feats,
            idx_means,
            idx_stds,
            grad_feats,
            grid,
        ],
        axis=0,
    ).astype(np.float32)
    return features


def _mad(x: np.ndarray) -> np.ndarray:
    med = np.median(x, axis=(1, 2))
    return np.median(np.abs(x - med[:, None, None]), axis=(1, 2)).astype(np.float32)


def extract_tabular_features_v3(patch: np.ndarray) -> np.ndarray:
    # Base stats (v2)
    means = patch.mean(axis=(1, 2))
    stds = patch.std(axis=(1, 2))
    qs = np.quantile(patch, [0.1, 0.25, 0.5, 0.75, 0.9], axis=(1, 2))  # (5, C)
    mins = patch.min(axis=(1, 2))
    maxs = patch.max(axis=(1, 2))
    mads = _mad(patch)
    iqr = (qs[3] - qs[1]).astype(np.float32)

    # Aspect handling
    aspect = patch[14]
    sin_aspect, cos_aspect = aspect_to_sin_cos(aspect)
    aspect_feats = np.array(
        [
            sin_aspect.mean(),
            cos_aspect.mean(),
            sin_aspect.std(),
            cos_aspect.std(),
        ],
        dtype=np.float32,
    )

    # Spectral indices
    eps = 1e-6
    red = patch[3]
    green = patch[2]
    nir = patch[7]
    swir1 = patch[10]
    swir2 = patch[11]

    ndvi = (nir - red) / (nir + red + eps)
    ndwi = (green - nir) / (green + nir + eps)
    ndmi = (nir - swir1) / (nir + swir1 + eps)
    nbr = (nir - swir2) / (nir + swir2 + eps)
    evi = 2.5 * (nir - red) / (nir + 6.0 * red - 7.5 * green + 1.0 + eps)
    savi = 1.5 * (nir - red) / (nir + red + 0.5 + eps)
    idx_stack = np.stack([ndvi, ndwi, ndmi, nbr, evi, savi], axis=0)
    idx_means = idx_stack.mean(axis=(1, 2))
    idx_stds = idx_stack.std(axis=(1, 2))
    idx_qs = np.quantile(idx_stack, [0.1, 0.5, 0.9], axis=(1, 2)).reshape(-1)

    # Band ratios (simple diversity signal)
    ratios = np.stack(
        [
            nir / (red + eps),
            swir1 / (nir + eps),
            swir2 / (nir + eps),
        ],
        axis=0,
    )
    ratio_means = ratios.mean(axis=(1, 2))
    ratio_stds = ratios.std(axis=(1, 2))

    # Texture on key bands + terrain
    grad_channels = np.array([3, 7, 10, 11, 12, 13], dtype=int)
    grad_feats = _mean_abs_grad(patch[grad_channels])

    # Terrain stats
    elev = patch[12]
    slope = patch[13]
    terrain = np.array(
        [
            elev.mean(),
            elev.std(),
            np.percentile(elev, 10),
            np.percentile(elev, 90),
            slope.mean(),
            slope.std(),
            np.percentile(slope, 10),
            np.percentile(slope, 90),
        ],
        dtype=np.float32,
    )

    # Grid pooling (v1 feature)
    grid = pooled_grid_features(patch)

    features = np.concatenate(
        [
            means,
            stds,
            qs.reshape(-1),
            mins,
            maxs,
            mads,
            iqr,
            aspect_feats,
            idx_means,
            idx_stds,
            idx_qs,
            ratio_means,
            ratio_stds,
            grad_feats,
            terrain,
            grid,
        ],
        axis=0,
    ).astype(np.float32)
    return features
