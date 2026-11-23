"""DeepCoRe modules: MCR, AFEM, MSCF (MSAFusion) and Branch."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AblationFlags:
    """Toggle individual DeepCoRe components for ablation studies."""

    use_mcr: bool = True
    use_afem: bool = True
    use_mscf: bool = True


class Attention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = dim ** -0.5
        self.fc1 = nn.Linear(dim, dim, bias=False)
        self.fc2 = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, dictionary, S):
        b1, n1, c1 = x.shape
        b2, n2, c2 = dictionary.shape
        S = S.unsqueeze(2)
        dictionary = dictionary * S
        q = self.fc1(x).reshape(b1, n1, 1, c1).permute(2, 0, 1, 3)
        k = self.fc2(dictionary).reshape(b2, n2, 1, c2).permute(2, 0, 1, 3)
        q, k = q[0], k[0]
        t = (q @ k.transpose(-2, -1)) * self.scale
        tavg = torch.mean(t, dim=2, keepdim=True)
        tatt = tavg.softmax(dim=1).repeat(1, 1, c1)
        return x * tatt


class AFEM(nn.Module):
    """Auxiliary Feature Enhancement Module."""

    def __init__(self, dim: int, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim)

    def forward(self, x, dictionary, S):
        return x + self.attn(self.norm1(x), dictionary, S).squeeze(1)


class MCR(nn.Module):
    """Multi-head Collaborative Representation module."""

    def __init__(self, dim: int = 2048, num_heads: int = 8, temp: float = 0.05):
        super().__init__()
        self.num_heads = num_heads
        self.fc_d_heads = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_heads)])
        self.temp = temp

    def forward(self, dictionary, tf):
        s_list, d_list = [], []
        for fc_d in self.fc_d_heads:
            fcd = fc_d(dictionary)
            sim = torch.matmul(tf, fcd.T)
            s = F.softmax(sim / self.temp, dim=1)
            d = torch.matmul(s, dictionary)
            s_list.append(s)
            d_list.append(d)
        return torch.stack(s_list).mean(0), torch.stack(d_list).mean(0)


class MSAFusion(nn.Module):
    """Multi-Stage Channel Fusion (MSCF)."""

    def __init__(self, dim: int, reduction: int = 8):
        super().__init__()
        self.dim = dim
        self.reduction = reduction
        self.se3 = self._make_se_branch()
        self.se4 = self._make_se_branch()
        self.se5 = self._make_se_branch()
        self.fusion = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 1, 1),
            nn.BatchNorm2d(dim),
        )

    def _make_se_branch(self):
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.dim, self.dim // self.reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dim // self.reduction, self.dim, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, h3, h4, h5):
        f3 = h3 * self.se3(h3)
        f4 = h4 * self.se4(h4)
        f5 = h5 * self.se5(h5)
        return self.fusion(f3 + f4 + f5)


class GroupingUnit(nn.Module):
    def __init__(self, num_classes: int, dim: int, ablation: AblationFlags | None = None):
        super().__init__()
        self.num_parts = num_classes
        self.num_classes = num_classes
        self.dim = dim
        self.ablation = ablation or AblationFlags()

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, dim))
        self.fcp2 = nn.Sequential(nn.Linear(dim, num_classes))
        self.afem1 = AFEM(dim=dim)
        self.afem2 = AFEM(dim=dim)
        self.afem3 = AFEM(dim=dim)
        self.mcr = MCR(dim=dim)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fusion = MSAFusion(self.dim)

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight, mode="fan_out")

    def forward(self, x, features):
        batch_size = features[0].size(0)
        dictionary = self.weight.contiguous().view(self.num_parts, self.dim)

        if self.ablation.use_mcr:
            s_x, d = self.mcr(dictionary, x)
        else:
            s_x = torch.ones((x.shape[0], dictionary.shape[0]), device=x.device) / dictionary.shape[0]
            d = x

        dict_batch = dictionary.unsqueeze(0).repeat(batch_size, 1, 1)
        if self.ablation.use_afem:
            h3 = self.afem1(features[0], dict_batch, s_x)
            h4 = self.afem2(features[1], dict_batch, s_x)
            h5 = self.afem3(features[2], dict_batch, s_x)
        else:
            h3, h4, h5 = features[0], features[1], features[2]

        spatial = int(math.sqrt(h3.shape[1]))
        h3 = h3.permute(0, 2, 1).reshape(batch_size, -1, spatial, spatial)
        h4 = h4.permute(0, 2, 1).reshape(batch_size, -1, spatial, spatial)
        h5 = h5.permute(0, 2, 1).reshape(batch_size, -1, spatial, spatial)

        if self.ablation.use_mscf:
            edge = self.fusion(h3, h4, h5)
        else:
            edge = h3 + h4 + h5

        edge = self.avg_pool(edge).flatten(1)
        p2 = self.fcp2(edge)
        return d, p2, [h3, h4, h5]


class Branch(nn.Module):
    """DeepCoRe classification branch on top of ResNet feature maps."""

    def __init__(self, num_classes: int, ablation: AblationFlags | None = None):
        super().__init__()
        self.dim = 2048
        self.fc_f = nn.Linear(2048, num_classes)
        self.conv3 = nn.Sequential(
            nn.Conv2d(self.dim // 4, self.dim, 4, 4),
            nn.BatchNorm2d(self.dim),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(self.dim // 2, self.dim, 2, 2),
            nn.BatchNorm2d(self.dim),
            nn.ReLU(),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(self.dim, self.dim, 1, 1),
            nn.BatchNorm2d(self.dim),
            nn.ReLU(),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.grouping = GroupingUnit(num_classes, self.dim, ablation=ablation)
        self.grouping.reset_parameters()
        self.apply(_init_weights)

    def forward(self, featuremap):
        e3, e4, e5 = featuremap
        x = self.avg_pool(e5).flatten(1)
        p1 = self.fc_f(x)
        f3 = self.conv3(e3)
        f4 = self.conv4(e4)
        f5 = self.conv5(e5)
        f3 = f3.permute(0, 2, 3, 1).reshape(f3.shape[0], f3.shape[2] * f3.shape[3], -1)
        f4 = f4.permute(0, 2, 3, 1).reshape(f4.shape[0], f4.shape[2] * f4.shape[3], -1)
        f5 = f5.permute(0, 2, 3, 1).reshape(f5.shape[0], f5.shape[2] * f5.shape[3], -1)
        d, p2, att = self.grouping(x, [f3, f4, f5])
        return x, p1, d, p2, att


def _init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm1d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
