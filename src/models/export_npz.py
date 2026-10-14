"""Convert raw training data into the .npz tile format used by dataset.py.

!! YOU MUST ADAPT `iter_samples()` to the on-disk layout of the dataset you downloaded. !!
(Kuro Siwo and Sen1Floods11 change layout between releases; read their README, or use Kuro Siwo's own
dataloader from https://github.com/Orion-AI-Lab/KuroSiwo and yield its tensors from here.)

Each yielded sample must be:
   before_vv_vh : float array (2,H,W) in dB   (pre-event SAR, VV and VH)
   after_vv_vh  : float array (2,H,W) in dB   (post-event SAR, SAME orbit track)
   label        : uint8 (H,W)

   python -m src.models.export_npz --src /path/to/KuroSiwo --dst data/tiles --val-frac 0.1 --holdout-substr nepal bangladesh
Tiles whose source path contains any --holdout-substr go to <dst>/test_holdout (the "never seen" evaluation set).
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np


def iter_samples(src: Path):
    """TODO: yield (name, before(2,H,W) dB, after(2,H,W) dB, label(H,W)) for every tile in the dataset."""
    raise NotImplementedError("Adapt iter_samples() to your downloaded dataset layout - see module docstring.")
    yield  # pragma: no cover


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", default="data/tiles")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--holdout-substr", nargs="*", default=[], help="source-path keywords reserved for the unseen test set")
    a = ap.parse_args()
    dst = Path(a.dst)
    for sub in ("train", "val", "test_holdout"):
        (dst / sub).mkdir(parents=True, exist_ok=True)
    n = 0
    for name, before, after, label in iter_samples(Path(a.src)):
        h = int(hashlib.md5(name.encode()).hexdigest(), 16) % 1000 / 1000
        if any(s.lower() in name.lower() for s in a.holdout_substr):
            sub = "test_holdout"
        else:
            sub = "val" if h < a.val_frac else "train"
        x = np.concatenate([before, after]).astype("float32")
        np.savez_compressed(dst / sub / f"{Path(name).stem}_{n:06d}.npz", x=x, y=label.astype("uint8"))
        n += 1
    print("wrote", n, "tiles to", dst)


if __name__ == "__main__":
    main()
