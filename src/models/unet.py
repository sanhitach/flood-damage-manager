"""U-Net for flood segmentation. Input: 4 channels = [before VV, before VH, after VV, after VH] (dB, normalised)."""
from __future__ import annotations

import segmentation_models_pytorch as smp


def build_model(pretrained: bool = False, in_channels: int = 4, encoder: str = "resnet34"):
    return smp.Unet(encoder_name=encoder, encoder_weights="imagenet" if pretrained else None,
                    in_channels=in_channels, classes=1)
