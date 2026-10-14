"""SAR helpers: dB conversion, nan-aware speckle filter, slope from DEM, normalisation for the U-Net."""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def to_db(lin: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.clip(lin, 1e-5, None))


def valid_mask(lin: np.ndarray) -> np.ndarray:
    """Valid where finite and > 0 in every band. Input (bands,H,W) or (H,W). Returns (H,W)."""
    v = np.isfinite(lin) & (lin > 0)
    return v.all(axis=0) if v.ndim == 3 else v


def boxcar(lin: np.ndarray, size: int = 5) -> np.ndarray:
    """Nan-aware boxcar speckle filter in LINEAR power (per band)."""
    if lin.ndim == 3:
        return np.stack([boxcar(b, size) for b in lin])
    ok = np.isfinite(lin) & (lin > 0)
    num = ndimage.uniform_filter(np.where(ok, lin, 0.0).astype("float64"), size, mode="nearest")
    den = ndimage.uniform_filter(ok.astype("float64"), size, mode="nearest")
    out = np.where(den > 0, num / np.maximum(den, 1e-9), np.nan)
    out[~ok] = np.nan
    return out.astype("float32")


def slope_deg(dem: np.ndarray, pixel_size: float) -> np.ndarray:
    d = np.where(np.isfinite(dem), dem, np.nanmin(dem) if np.isfinite(dem).any() else 0.0)
    gy, gx = np.gradient(d, pixel_size)
    return np.degrees(np.arctan(np.hypot(gx, gy))).astype("float32")


def normalize_db(x: np.ndarray) -> np.ndarray:
    """Clip dB to [-35, 5] and scale to roughly [-2, 2]. Used identically in training and inference."""
    x = np.nan_to_num(x, nan=-35.0)
    return ((np.clip(x, -35.0, 5.0) + 15.0) / 10.0).astype("float32")
