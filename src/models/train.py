"""Train the U-Net.
   python -m src.models.train --train data/tiles/train --val data/tiles/val --epochs 25 --out models/unet_flood.pt
Run on a GPU (Google Colab free tier is enough). Keep Himalayan test scenes OUT of train/val."""
from __future__ import annotations

import argparse
import time

import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader

from src.models.dataset import NpzTiles
from src.models.evaluate import metrics
from src.models.unet import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--positive", type=int, nargs="+", default=[2])
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--pretrained", action="store_true", help="ImageNet encoder weights (downloads on first use)")
    ap.add_argument("--out", default="models/unet_flood.pt")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr = DataLoader(NpzTiles(a.train, True, tuple(a.positive)), batch_size=a.batch, shuffle=True, num_workers=2, drop_last=True)
    va = DataLoader(NpzTiles(a.val, False, tuple(a.positive)), batch_size=a.batch, num_workers=2)
    model = build_model(a.pretrained).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    dice = smp.losses.DiceLoss(mode="binary", from_logits=True)
    bce = torch.nn.BCEWithLogitsLoss()
    best = -1.0
    for ep in range(a.epochs):
        model.train()
        t0, tot = time.time(), 0.0
        for x, y in tr:
            x, y = x.to(dev), y.to(dev)
            logits = model(x)
            loss = bce(logits, y) + dice(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        model.eval()
        P, T = [], []
        with torch.no_grad():
            for x, y in va:
                P.append((torch.sigmoid(model(x.to(dev))).cpu() > 0.5).numpy().astype("uint8"))
                T.append(y.numpy().astype("uint8"))
        import numpy as np
        m = metrics(np.concatenate(P), np.concatenate(T))
        print(f"epoch {ep+1:02d} loss {tot/len(tr):.4f} val IoU {m['iou']:.3f} F1 {m['f1']:.3f} ({time.time()-t0:.0f}s)")
        if m["iou"] > best:
            best = m["iou"]
            torch.save(model.state_dict(), a.out)
    print("best val IoU", best, "->", a.out)


if __name__ == "__main__":
    main()
