"""Evaluate a trained model on HELD-OUT tiles (e.g. Himalayan / South-Asian scenes never used in training).
   python -m src.models.evaluate --weights models/unet_flood.pt --data data/tiles/test_himalaya --out outputs/eval_unet.json
Also reports the log-ratio baseline when --baseline is given (needs the 'raw' dB tiles: it thresholds after-before)."""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models.dataset import NpzTiles
from src.models.unet import build_model


def metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    tp = float(((pred == 1) & (target == 1)).sum())
    fp = float(((pred == 1) & (target == 0)).sum())
    fn = float(((pred == 0) & (target == 1)).sum())
    iou = tp / max(tp + fp + fn, 1)
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"iou": iou, "precision": prec, "recall": rec, "f1": 2 * prec * rec / max(prec + rec, 1e-9)}


@torch.no_grad()
def run_eval(weights, data, positive=(2,), device=None, threshold=0.5, batch=8):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()
    ds = NpzTiles(data, augment=False, positive_labels=positive, crop=256)
    P, T, B = [], [], []
    for x, y in DataLoader(ds, batch_size=batch):
        P.append((torch.sigmoid(model(x.to(device))).cpu().numpy() > threshold).astype("uint8"))
        T.append(y.numpy().astype("uint8"))
        # baseline on the same tiles: VV drop of > 3 dB and after < -17 dB (undo normalisation: dB = x*10 - 15)
        xb = x.numpy() * 10 - 15
        B.append(((xb[:, 2:3] - xb[:, 0:1] < -3) & (xb[:, 2:3] < -17)).astype("uint8"))
    P, T, B = np.concatenate(P), np.concatenate(T), np.concatenate(B)
    return {"n_tiles": len(ds), "unet": metrics(P, T), "baseline_logratio": metrics(B, T)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True, help="folder of held-out .npz tiles")
    ap.add_argument("--positive", type=int, nargs="+", default=[2])
    ap.add_argument("--out", default="outputs/eval.json")
    a = ap.parse_args()
    res = run_eval(a.weights, a.data, tuple(a.positive))
    print(json.dumps(res, indent=2))
    json.dump(res, open(a.out, "w"), indent=2)
