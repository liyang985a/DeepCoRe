"""ResNet-50 backbone used by DeepCoRe."""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torchvision.models as models


class ModifiedResNet50(nn.Module):
    """ResNet-50 feature extractor returning multi-stage maps for MSCF/AFEM."""

    def __init__(self, pretrained_path: str | None = None, freeze_early: bool = True):
        super().__init__()
        original_model = models.resnet50(weights=None)

        self.conv1 = original_model.conv1
        self.bn1 = original_model.bn1
        self.relu = original_model.relu
        self.maxpool = original_model.maxpool
        self.layer1 = original_model.layer1
        self.layer2 = original_model.layer2
        self.layer3 = original_model.layer3
        self.layer4 = original_model.layer4

        if freeze_early:
            for module in (self.conv1, self.bn1, self.layer1):
                for param in module.parameters():
                    param.requires_grad = False

        if pretrained_path and os.path.exists(pretrained_path):
            self.load_pretrained_weights(pretrained_path)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        features_layer3 = self.layer3(x)
        features_layer4 = self.layer4(features_layer3)
        # stage2 / stage3 / stage4 feature maps
        return [x, features_layer3, features_layer4]

    def load_pretrained_weights(self, weight_path: str) -> None:
        weights = torch.load(weight_path, map_location="cpu")
        if isinstance(weights, dict) and "state_dict" in weights:
            weights = weights["state_dict"]
        missing, unexpected = self.load_state_dict(weights, strict=False)
        print(f"Loaded backbone weights from {weight_path}")
        if missing:
            print(f"  missing keys: {len(missing)}")
        if unexpected:
            print(f"  unexpected keys: {len(unexpected)}")
