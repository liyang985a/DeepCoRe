"""Training losses for DeepCoRe."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LOSS_C(nn.Module):
    """Cosine alignment loss between backbone features and reconstructed features.

    Note for manuscript revision: this is cosine alignment (no negative pairs),
    not a standard contrastive InfoNCE loss.
    """

    def forward(self, g_s, g_t):
        similarity = torch.cosine_similarity(g_s, g_t, dim=1)
        return 1 - similarity.mean()


class LOSS_K(nn.Module):
    """Inter-layer semantic consistency regularizer used by AFEM."""

    def forward(self, features):
        loss = 0.0
        for i in range(1, len(features)):
            loss = loss + self.at_loss(features[i - 1], features[i].detach())
        return loss

    def at_loss(self, f_s, f_t):
        return (self.at(f_s) - self.at(f_t)).pow(2).mean()

    def at(self, f):
        return f.pow(2).mean(2)


class FocalLoss(nn.Module):
    """Softmax focal loss: -w_y (1-p_y)**gamma log(p_y).

    Mean reduction averages weighted sample losses over the batch. In
    particular, weights must not enter the probability p_y.
    """

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None, reduction: str = "mean"):
        super().__init__()
        if gamma < 0 or reduction not in ("mean", "sum", "none"):
            raise ValueError("Expected nonnegative gamma and mean/sum/none reduction")
        self.gamma = gamma
        self.register_buffer("weight", weight)
        self.reduction = reduction

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        if self.weight is not None:
            loss = loss * self.weight[targets]
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def clip_gradient(optimizer, grad_clip: float = 0.5) -> None:
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)
