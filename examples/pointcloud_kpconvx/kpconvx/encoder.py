# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
# KPConvX project: encoder.py  (NEW — not from apple/ml-kpconvx)
#
# This file is an eb_jepa original: it wraps the vendored KPConv operators
# (KPConvBlock / KPConvResidualBlock) with a pure-torch neighborhood pyramid
# (grid subsampling + radius neighbor search via torch.cdist) so the encoder
# runs WITHOUT pykeops or compiled C++ extensions. It exposes the eb_jepa
# encoder contract: .represent(x) -> [B, D], .out_dim, forward == represent.

import math
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from .kpconv_blocks import KPConvBlock, KPConvResidualBlock
from .generic_blocks import GlobalAverageBlock, UnaryBlock, local_maxpool


# --------------------------------------------------------------------------- #
#  Pure-torch neighborhood pyramid  (replaces utils/torch_pyramid.py + pykeops)
# --------------------------------------------------------------------------- #

@torch.no_grad()
def grid_subsample(points: torch.Tensor, lengths: torch.Tensor, dl: float):
    """Grid subsample a packed point cloud.

    Args:
        points:  (N, 3) packed point coordinates.
        lengths: (B,) number of points per cloud in the pack.
        dl:      grid cell size.

    Returns:
        sub_points:  (M, 3) subsampled packed points.
        sub_lengths: (B,) number of points per cloud after subsampling.
    """
    if dl <= 0:
        return points, lengths
    device = points.device
    sub_pts_list = []
    sub_lens = []
    i0 = 0
    for b in range(lengths.shape[0]):
        n = int(lengths[b].item())
        pts = points[i0:i0 + n]  # (n, 3)
        i0 += n
        # Quantize to grid cells, keep first point per cell
        keys = torch.floor(pts / dl).long()
        # Encode 3D grid key to a single integer (offset by +1 to avoid -0 issues)
        keys = keys + keys.min(dim=0).values.abs() + 1  # shift to non-negative
        # Use a unique hash: keys are (n,3) -> encode as single int
        mx = keys.max(dim=0).values + 1
        flat_keys = keys[:, 0] * (mx[1] * mx[2]) + keys[:, 1] * mx[2] + keys[:, 2]
        _, inv, counts = torch.unique(flat_keys, return_inverse=True, return_counts=True)
        # Keep the first occurrence of each unique key
        seen = torch.zeros(flat_keys.shape[0], dtype=torch.bool, device=device)
        first_idx = []
        # Efficient: sort by inv, take first of each group
        order = torch.argsort(inv, stable=True)
        sorted_inv = inv[order]
        mask = torch.ones_like(sorted_inv, dtype=torch.bool)
        mask[1:] = sorted_inv[1:] != sorted_inv[:-1]
        first_idx = order[mask]
        sub_pts_list.append(pts[first_idx])
        sub_lens.append(first_idx.shape[0])
    if sub_pts_list:
        sub_points = torch.cat(sub_pts_list, dim=0)
        sub_lengths = torch.tensor(sub_lens, device=device, dtype=lengths.dtype)
    else:
        sub_points = points.new_zeros((0, 3))
        sub_lengths = lengths.new_zeros((0,))
    return sub_points, sub_lengths


@torch.no_grad()
def radius_neighbors(q_pts: torch.Tensor, s_pts: torch.Tensor,
                     q_lengths: torch.Tensor, s_lengths: torch.Tensor,
                     radius: float, max_neighbors: int = 64):
    """Vectorized radius neighbor search in packed mode using torch.cdist.

    Args:
        q_pts:  (M, 3) query points.
        s_pts:  (N, 3) support points.
        q_lengths: (B,) query pack lengths.
        s_lengths: (B,) support pack lengths.
        radius: search radius.
        max_neighbors: cap on neighbors per query (pads with N as shadow index).

    Returns:
        neighbor_indices: (M, K) LongTensor — indices into the N support points.
                          Out-of-range neighbors are set to N (shadow padding).
    """
    device = q_pts.device
    K = max_neighbors
    M = q_pts.shape[0]
    N = s_pts.shape[0]
    neighbor_indices = torch.full((M, K), N, dtype=torch.long, device=device)
    BIG = float("inf")

    qi0 = si0 = 0
    for b in range(q_lengths.shape[0]):
        qn = int(q_lengths[b].item())
        sn = int(s_lengths[b].item())
        if qn == 0 or sn == 0:
            qi0 += qn
            si0 += sn
            continue
        qp = q_pts[qi0:qi0 + qn]   # (qn, 3)
        sp = s_pts[si0:si0 + sn]   # (sn, 3)
        # Pairwise distances (qn, sn) — fully vectorized, no Python loop
        dists = torch.cdist(qp, sp)              # (qn, sn)
        # Set out-of-radius distances to +inf so topk ignores them
        dists = torch.where(dists <= radius, dists, torch.full_like(dists, BIG))
        # If fewer than K neighbors, pad with inf
        if sn < K:
            pad = torch.full((qn, K - sn), BIG, device=device)
            dists = torch.cat([dists, pad], dim=1)           # (qn, K)
            idx = torch.arange(sn, device=device).repeat(qn, 1)
            idx = torch.cat([idx, torch.full((qn, K - sn), sn, dtype=torch.long, device=device)], dim=1)
        else:
            # Top-K nearest (smallest distances) — vectorized across all queries
            topk_dists, topk_idx = torch.topk(dists, K, dim=1, largest=False)  # (qn, K)
            idx = topk_idx
        # Where distance is inf, set index to sn (shadow)
        mask = topk_dists >= BIG if sn >= K else (dists >= BIG)
        idx = torch.where(mask, torch.full_like(idx, sn), idx)
        neighbor_indices[qi0:qi0 + qn] = idx + si0
        qi0 += qn
        si0 += sn
    return neighbor_indices


@torch.no_grad()
def build_pyramid(points: torch.Tensor, lengths: torch.Tensor,
                  num_layers: int, sub_size: float, search_radius: float,
                  neighbor_limits: List[int]):
    """Build a multi-scale neighborhood pyramid (pure-torch, no pykeops).

    Returns a dict with keys: points, lengths, neighbors, pools (lists per layer).
    """
    pyramid = {
        "points": [points],
        "lengths": [lengths],
        "neighbors": [],
        "pools": [],
    }
    cur_pts = points
    cur_lens = lengths
    cur_sub = sub_size
    cur_radius = search_radius
    for i in range(num_layers):
        # Subsample (except layer 0 which is the input)
        if i > 0:
            cur_pts, cur_lens = grid_subsample(cur_pts, cur_lens, cur_sub)
            pyramid["points"].append(cur_pts)
            pyramid["lengths"].append(cur_lens)
            cur_sub *= 2.0
        # Neighbors for convolution at this layer
        neighb = radius_neighbors(cur_pts, cur_pts, cur_lens, cur_lens,
                                  cur_radius, neighbor_limits[i])
        pyramid["neighbors"].append(neighb)
        # Pooling indices (from current layer to next layer's points)
        if i < num_layers - 1:
            sub_pts_next, sub_lens_next = grid_subsample(cur_pts, cur_lens, cur_sub)
            pool_inds = radius_neighbors(sub_pts_next, cur_pts, sub_lens_next,
                                         cur_lens, cur_radius, neighbor_limits[i])
            pyramid["pools"].append(pool_inds)
            pyramid["points_next"] = sub_pts_next  # for reference
            pyramid["lengths_next"] = sub_lens_next
        cur_radius *= 2.0
    return pyramid


# --------------------------------------------------------------------------- #
#  KPConvX classification encoder
# --------------------------------------------------------------------------- #

class KPConvXEncoder(nn.Module):
    """KPConvX classification-style encoder for point clouds [B, 3, N] -> [B, D].

    Wraps the vendored KPConv blocks (KPConvBlock + KPConvResidualBlock) with a
    multi-scale neighborhood pyramid (pure-torch grid subsampling + radius search)
    and a final GlobalAverageBlock, matching the KPCNN classification architecture
    from apple/ml-kpconvx.

    The final GlobalAverageBlock makes the global feature permutation-INVARIANT
    (same role as PointNet's max-pool). Rotation invariance is still LEARNED from
    the augmented views.
    """

    def __init__(self,
                 in_channels: int = 3,
                 out_dim: int = 256,
                 # architecture
                 layer_blocks: List[int] = (1, 1, 1, 1),
                 first_features_dim: int = 32,
                 shell_sizes: List[int] = (1, 3, 3),
                 # KPConv geometry
                 conv_radius: float = 1.5,
                 first_subsampling_dl: float = 0.02,
                 fixed_kernel_points: str = "center",
                 kp_influence: str = "linear",
                 kp_aggregation: str = "sum",
                 dimension: int = 3,
                 # norm / training
                 norm: str = "batch",
                 bn_momentum: float = 0.02,
                 # neighborhood
                 neighbor_limits: Optional[List[int]] = None,
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.out_dim = out_dim
        self.dimension = dimension
        self.shell_sizes = list(shell_sizes)
        self.conv_radius = conv_radius
        self.first_subsampling_dl = first_subsampling_dl
        self.layer_blocks = list(layer_blocks)
        self.num_layers = len(self.layer_blocks)

        if neighbor_limits is None:
            neighbor_limits = [64] * self.num_layers
        self.neighbor_limits = list(neighbor_limits)

        # Build the block sequence (stem + per-layer residual blocks + strided pool)
        # Mirrors KPCNN: stem KPConvBlock, then per layer: residual blocks + strided pool
        self.block_ops = nn.ModuleList()

        r = first_subsampling_dl * conv_radius
        in_dim = in_channels
        out_dim_layer = first_features_dim

        for layer_idx in range(self.num_layers):
            n_blocks = self.layer_blocks[layer_idx]
            if layer_idx == 0:
                # Stem: simple KPConvBlock
                self.block_ops.append(KPConvBlock(
                    in_dim, out_dim_layer, self.shell_sizes, r, r,
                    influence_mode=kp_influence, aggregation_mode=kp_aggregation,
                    dimension=dimension, norm_type=norm, bn_momentum=bn_momentum,
                ))
                in_dim = out_dim_layer
                n_blocks_remaining = n_blocks - 1
            else:
                n_blocks_remaining = n_blocks

            # Residual blocks within the layer
            for _ in range(n_blocks_remaining):
                self.block_ops.append(KPConvResidualBlock(
                    in_dim, out_dim_layer, self.shell_sizes, r, r,
                    influence_mode=kp_influence, aggregation_mode=kp_aggregation,
                    dimension=dimension, norm_type=norm, bn_momentum=bn_momentum,
                    strided=False,
                ))

            # Strided pooling block to next layer (except last layer)
            if layer_idx < self.num_layers - 1:
                next_dim = out_dim_layer * 2
                self.block_ops.append(KPConvResidualBlock(
                    in_dim, next_dim, self.shell_sizes, r, r,
                    influence_mode=kp_influence, aggregation_mode=kp_aggregation,
                    dimension=dimension, norm_type=norm, bn_momentum=bn_momentum,
                    strided=True,
                ))
                in_dim = next_dim
                out_dim_layer = next_dim
                r *= 2.0
            else:
                in_dim = out_dim_layer

        # Global average pool + head MLP -> out_dim (no classifier logits, SSL)
        self.global_pool = GlobalAverageBlock()
        self.head = UnaryBlock(in_dim, out_dim, norm_type="none", bn_momentum=0.0)

    def _batch_to_pack(self, x: torch.Tensor):
        """Convert [B, C, N] -> packed (Ntot, C) + lengths (B,).

        Also handles the case where x is [B, N, C].
        """
        if x.dim() != 3:
            raise ValueError(f"expected [B, C, N] point cloud tensor, got {tuple(x.shape)}")
        B, C, N = x.shape
        if C != self.in_channels and N == self.in_channels:
            # Input is [B, N, C] — transpose to [B, C, N]
            x = x.transpose(1, 2)
            B, C, N = x.shape
        # Flatten to packed (B*N, C)
        points = x.permute(0, 2, 1).reshape(B * N, C)  # (B*N, C)
        lengths = torch.full((B,), N, dtype=torch.long, device=x.device)
        return points, lengths

    def represent(self, x: torch.Tensor) -> torch.Tensor:
        """Map [B, 3, N] -> [B, out_dim] global feature."""
        B = x.shape[0]
        points, lengths = self._batch_to_pack(x)  # (Ntot, 3), (B,)

        # Build the neighborhood pyramid
        pyramid = build_pyramid(
            points, lengths,
            num_layers=self.num_layers,
            sub_size=self.first_subsampling_dl,
            search_radius=self.first_subsampling_dl * self.conv_radius,
            neighbor_limits=self.neighbor_limits,
        )

        # Forward through the block sequence
        # Input features: the point coordinates themselves (like KPCNN in_features_dim)
        feats = points  # (Ntot, in_channels=3)

        block_idx = 0
        layer_pts = pyramid["points"][0]
        layer_lens = pyramid["lengths"][0]
        layer_neighb = pyramid["neighbors"][0]
        pool_idx = 0

        for layer_idx in range(self.num_layers):
            n_blocks = self.layer_blocks[layer_idx]
            if layer_idx == 0:
                # Stem block
                block = self.block_ops[block_idx]
                feats = block(layer_pts, layer_pts, feats, layer_neighb)
                block_idx += 1
                n_blocks_remaining = n_blocks - 1
            else:
                n_blocks_remaining = n_blocks

            # Residual blocks
            for _ in range(n_blocks_remaining):
                block = self.block_ops[block_idx]
                feats = block(layer_pts, layer_pts, feats, layer_neighb)
                block_idx += 1

            # Strided pool to next layer
            if layer_idx < self.num_layers - 1:
                pool_block = self.block_ops[block_idx]
                next_pts = pyramid["points"][layer_idx + 1]
                next_lens = pyramid["lengths"][layer_idx + 1]
                pool_inds = pyramid["pools"][pool_idx]
                pool_idx += 1
                feats = pool_block(next_pts, layer_pts, feats, pool_inds)
                block_idx += 1
                layer_pts = next_pts
                layer_lens = next_lens
                layer_neighb = pyramid["neighbors"][layer_idx + 1]

        # Global average pool -> [B, C]
        global_feat = self.global_pool(feats, layer_lens)
        # Head MLP -> [B, out_dim]
        global_feat = self.head(global_feat)
        return global_feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.represent(x)
