"""Tile dataset. Each tile is an .npz with
   x : float32 (4, H, W)  [before VV, before VH, after VV, after VH] in dB (NOT normalised)
   y : uint8   (H, W)     label map
Kuro Siwo labels: 0 = no water, 1 = permanent water, 2 = flood  ->  positive_labels=(2,)
Sen1Floods11 labels: 1 = water -> positive_labels=(1,)
See src/models/export_npz.py for how to create these files from the raw datasets."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.preprocess.sar import normalize_db


class NpzTiles(Dataset):
    def __init__(self, root, augment=False, positive_labels=(2,), crop=256):
        self.files = sorted(Path(root).glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No .npz tiles in {root}")
        self.augment, self.pos, self.crop = augment, positive_labels, crop

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        d = np.load(self.files[i])
        x, y = normalize_db(d["x"]), np.isin(d["y"], self.pos).astype("float32")
        _, H, W = x.shape
        c = min(self.crop, H, W)
        r0 = np.random.randint(0, H - c + 1) if self.augment else (H - c) // 2
        c0 = np.random.randint(0, W - c + 1) if self.augment else (W - c) // 2
        x, y = x[:, r0:r0 + c, c0:c0 + c], y[r0:r0 + c, c0:c0 + c]
        if self.augment:
            k = np.random.randint(4)
            x, y = np.rot90(x, k, (1, 2)), np.rot90(y, k)
            if np.random.rand() < 0.5:
                x, y = x[:, :, ::-1], y[:, ::-1]
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(y))[None]
