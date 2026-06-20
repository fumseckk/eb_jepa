"""Point-JEPA helper modules: EMA, Predictor, sampling strategies."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


# ============================================================================
# EMA (Exponential Moving Average) Teacher
# ============================================================================
class EMA(nn.Module):
    """Maintains an exponential moving average copy of the student model."""

    def __init__(self, student, tau_min=0.99, tau_max=0.9998, tau_steps=1000):
        super().__init__()
        self.student = student
        self.ema_model = self._deepcopy_module(student)
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.tau_steps = tau_steps
        self.step_count = 0

    def _deepcopy_module(self, model):
        """Create a deep copy of the model."""
        import copy
        return copy.deepcopy(model)

    def update(self):
        """Update EMA parameters."""
        with torch.no_grad():
            tau = self.tau_min + (self.tau_max - self.tau_min) * (
                1 - (self.step_count / max(self.tau_steps, 1))
            )
            tau = max(self.tau_min, min(tau, self.tau_max))
            for ema_p, student_p in zip(
                self.ema_model.parameters(), self.student.parameters()
            ):
                ema_p.data.mul_(tau).add_(student_p.data, alpha=1 - tau)
        self.step_count += 1

    def forward(self, x):
        return self.ema_model(x)


# ============================================================================
# Predictor: Transformer that maps context to target embeddings
# ============================================================================
class Predictor(nn.Module):
    """Predicts target embeddings from context embeddings using a Transformer."""

    def __init__(
        self,
        embed_dim: int = 384,
        depth: int = 6,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    drop_path=dpr[i],
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self, context: torch.Tensor, context_pos: torch.Tensor, target_pos: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            context: [B, T_ctx, D] context embeddings
            context_pos: [B, T_ctx, D] positional embeddings for context
            target_pos: [B, T_tgt, D] positional embeddings for targets

        Returns:
            predictions: [B, T_tgt, D] predicted target embeddings
        """
        B, T_ctx, D = context.shape
        B, T_tgt, D = target_pos.shape

        # Cross-attention: use target positions to attend over context
        x = target_pos  # [B, T_tgt, D]

        for block in self.blocks:
            x = block(x, context, context_pos)

        x = self.norm(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with cross-attention to context."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm_context = nn.LayerNorm(embed_dim)

        # Cross-attention: query from target positions, key/value from context
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=attn_dropout, batch_first=True
        )

        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, embed_dim),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(
        self, x: torch.Tensor, context: torch.Tensor, context_pos: torch.Tensor
    ) -> torch.Tensor:
        # x: [B, T_tgt, D] (target positions)
        # context: [B, T_ctx, D]
        # context_pos: [B, T_ctx, D]

        # Inject context positional encoding
        context_with_pos = context + context_pos

        # Cross-attention
        q = self.norm1(x)
        k = self.norm_context(context_with_pos)
        v = context_with_pos
        attn_out, _ = self.attn(q, k, v)
        x = x + self.drop_path(attn_out)

        # MLP
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class DropPath(nn.Module):
    """Stochastic depth (drop path)."""

    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.p == 0 or not self.training:
            return x
        keep = 1 - self.p
        mask = torch.rand(x.shape[0], *([1] * (x.ndim - 1)), device=x.device) < keep
        return x * mask / keep


# ============================================================================
# Sampling: split patches into context (visible) and target (masked)
# ============================================================================
class TargetSampler(nn.Module):
    """Sample which patches are 'targets' (masked regions)."""

    def __init__(
        self,
        num_targets: int = 4,
        target_ratio: Tuple[float, float] = (0.15, 0.2),
    ):
        super().__init__()
        self.num_targets = num_targets
        self.target_ratio_min, self.target_ratio_max = target_ratio

    def forward(self, tokens: torch.Tensor) -> Tuple[list, list]:
        """
        Args:
            tokens: [B, T, D] all patches

        Returns:
            target_tokens_list: list of [B, n_tgt, D] target samples (may have different sizes)
            target_indices_list: list of indices for each target sample
        """
        B, T, D = tokens.shape
        device = tokens.device

        target_list = []
        indices_list = []

        # Sample target ratio once and use for all num_targets samples
        # This ensures consistent target sizes across samples
        ratio = torch.empty(1).uniform_(
            self.target_ratio_min, self.target_ratio_max
        ).item()
        n_targets = max(1, int(T * ratio))

        for m in range(self.num_targets):
            # Random sample patches as targets (different sample each iteration)
            perm = torch.randperm(T, device=device)
            target_idx = perm[:n_targets]

            target_list.append(tokens[:, target_idx])  # [B, n_targets, D]
            indices_list.append(target_idx)

        return target_list, indices_list


class ContextSampler(nn.Module):
    """Sample which patches form the 'context' (visible, unmasked)."""

    def __init__(
        self,
        context_ratio: Tuple[float, float] = (0.4, 0.75),
    ):
        super().__init__()
        self.context_ratio_min, self.context_ratio_max = context_ratio

    def forward(
        self, tokens: torch.Tensor, target_indices_list
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            tokens: [B, T, D] all patches
            target_indices_list: list of target indices per sample

        Returns:
            context_tokens: [B, T_ctx, D]
            context_indices: [B, T_ctx]
        """
        B, T, D = tokens.shape
        device = tokens.device

        ratio = torch.empty(1).uniform_(
            self.context_ratio_min, self.context_ratio_max
        ).item()
        n_context = max(1, int(T * ratio))

        # Context = all patches except targets (or random subset)
        context_list = []
        for b in range(B):
            # Simple: just take first n_context patches that aren't targets
            all_idx = set(range(T))
            target_set = set(target_indices_list[0].cpu().numpy())  # use first target set
            context_idx = list(all_idx - target_set)
            context_idx = context_idx[:n_context]
            context_list.append(tokens[b, context_idx])

        context_tokens = torch.stack(context_list)  # [B, T_ctx, D]
        return context_tokens
