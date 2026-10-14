"""Baseline flood/debris detector: log-ratio change detection with terrain masking.

Classes in the output mask: 0 = none, 1 = new open water (flood / ponded water), 2 = debris / disturbed corridor.
Same-orbit-track before/after images are REQUIRED (see README).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from src.preprocess import sar


def remove_small(mask: np.ndarray, min_pixels: int) -> np.ndarray:
    lab, n = ndimage.label(mask)
    if n == 0:
        return mask
    sizes = np.bincount(lab.ravel())
    keep = sizes >= min_pixels
    keep[0] = False
    return keep[lab]


def detect(before_lin, after_lin, slope, pixel_size, cfg):
    """before_lin/after_lin: (2,H,W) linear sigma0 (VV,VH). Returns (mask uint8, change_db float32 (H,W))."""
    k = int(cfg["speckle_window"])
    b = sar.to_db(sar.boxcar(before_lin, k))
    a = sar.to_db(sar.boxcar(after_lin, k))
    d = a - b
    valid = np.isfinite(d).all(axis=0)
    terrain_ok = slope < cfg["max_slope_deg"]

    water = (d[0] < -cfg["drop_db"]) & (a[0] < cfg["water_db"]) & (b[0] > cfg["water_db"])
    water &= valid & terrain_ok
    min_px = max(1, int(cfg["min_area_m2"] / pixel_size**2))
    water = remove_small(ndimage.binary_opening(water, iterations=1), min_px)

    corridor_px = max(1, int(cfg["debris_corridor_m"] / pixel_size))
    near_water = ndimage.binary_dilation(water, iterations=corridor_px)
    rise = np.nanmean(d, axis=0) > cfg["rise_db"]
    debris = rise & near_water & ~water & valid & terrain_ok
    debris = remove_small(ndimage.binary_opening(debris, iterations=1), min_px)

    mask = np.zeros(water.shape, dtype="uint8")
    mask[water] = 1
    mask[debris] = 2
    change = np.where(valid, np.nanmean(d, axis=0), np.nan).astype("float32")
    return mask, change


def finalize_mask(water, debris_mask, valid, terrain_ok, pixel_size, cfg):
    """Used by the U-Net path: combine model water with heuristic debris."""
    min_px = max(1, int(cfg["min_area_m2"] / pixel_size**2))
    water = remove_small(water & valid & terrain_ok, min_px)
    out = np.zeros(water.shape, dtype="uint8")
    out[debris_mask & ~water] = 2
    out[water] = 1
    return out
