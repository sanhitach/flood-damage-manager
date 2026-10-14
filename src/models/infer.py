"""Sliding-window inference on a whole scene."""
from __future__ import annotations

import numpy as np
import torch

from src.models.unet import build_model


@torch.no_grad()
def predict(weights, x: np.ndarray, tile: int = 256, overlap: int = 64, device: str | None = None, batch: int = 8):
    """x: (4,H,W) NORMALISED channels. Returns flood probability (H,W) float32."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()
    C, H, W = x.shape
    stride = tile - overlap
    Hp = max(tile, int(np.ceil((H - tile) / stride)) * stride + tile)
    Wp = max(tile, int(np.ceil((W - tile) / stride)) * stride + tile)
    xp = np.pad(x, ((0, 0), (0, Hp - H), (0, Wp - W)), mode="edge")
    acc, cnt = np.zeros((Hp, Wp), "float32"), np.zeros((Hp, Wp), "float32")
    wins = [(r, c) for r in range(0, Hp - tile + 1, stride) for c in range(0, Wp - tile + 1, stride)]
    for i in range(0, len(wins), batch):
        chunk = wins[i:i + batch]
        t = torch.from_numpy(np.stack([xp[:, r:r + tile, c:c + tile] for r, c in chunk])).to(device)
        p = torch.sigmoid(model(t))[:, 0].cpu().numpy()
        for (r, c), pp in zip(chunk, p):
            acc[r:r + tile, c:c + tile] += pp
            cnt[r:r + tile, c:c + tile] += 1
    return (acc / np.maximum(cnt, 1))[:H, :W]
