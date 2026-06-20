"""PointCloud — Multi-view Point-JEPA (predictive learning with rotation invariance).

Research question: can a multi-view masked predictive objective learn a rotation-invariant
shape representation on point clouds, and how does probe accuracy degrade as we
demand more rotation invariance (none -> z -> SO(3))?

The objective is Multi-view JEPA: take two independent augmented views of the same shape
with different rotations. Mask patches in view1 (student's context), keep view2 fully
visible (teacher target). The student encodes unmasked patches of view1 and predicts
the teacher's encoding of the SAME patches but from the differently-rotated view2.
This forces rotation invariance: the student must learn which features are invariant
across rotations to predict the rotated view from the unrotated one.

The student and teacher share weights via EMA, predictor maps context -> target embeddings.

Run:  python -m examples.pointcloud.main --fname examples/pointcloud/cfgs/train.yaml
"""
import os
import sys
from dataclasses import asdict

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from eb_jepa.datasets.pointcloud.dataset import PointCloudConfig, make_loader
from eb_jepa.training_utils import setup_wandb

# Reuse the eb_jepa core — DO NOT reimplement these:
#   eb_jepa.architectures: Projector (MLP from a '256-512-128'-style spec string)
#   eb_jepa.losses:        VICRegLoss (invariance + variance + covariance)


# --------------------------------------------------------------------------- #
# 1) ENCODER  — # TODO
# --------------------------------------------------------------------------- #
def build_encoder(cfg):
    in_channels     = int(getattr(cfg, "in_channels", 3))
    out_dim         = int(getattr(cfg, "out_dim", 1024))
    n_centers       = int(getattr(cfg, "n_centers", 64))
    k_neighbors     = int(getattr(cfg, "k_neighbors", 32))
    d_model         = int(getattr(cfg, "d_model", 384))
    n_heads         = int(getattr(cfg, "n_heads", 6))
    n_layers        = int(getattr(cfg, "n_layers", 12))
    drop_path_rate  = float(getattr(cfg, "drop_path_rate", 0.25))
    attn_dropout    = float(getattr(cfg, "attn_dropout", 0.05))

    # Stochastic depth schedule (linearly increasing per layer, as in the paper)
    dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]

    # ------------------------------------------------------------------ #
    # Drop-path (stochastic depth) — scales residual to zero for a random
    # subset of samples in each batch, independent per layer.
    # ------------------------------------------------------------------ #
    class DropPath(nn.Module):
        def __init__(self, p: float):
            super().__init__()
            self.p = p

        def forward(self, x):
            if self.p == 0. or not self.training:
                return x
            keep = 1 - self.p
            mask = torch.rand(x.shape[0], *([1] * (x.ndim - 1)), device=x.device) < keep
            return x * mask / keep

    # ------------------------------------------------------------------ #
    # Transformer block with per-layer positional encoding injected into
    # Q and K (following Point-JEPA / BEiT-style "pos at every layer").
    # ------------------------------------------------------------------ #
    class TransformerBlock(nn.Module):
        def __init__(self, drop_path_p: float):
            super().__init__()
            self.norm1   = nn.LayerNorm(d_model)
            self.attn    = nn.MultiheadAttention(
                d_model, n_heads, dropout=attn_dropout, batch_first=True
            )
            self.dp1     = DropPath(drop_path_p)
            self.norm2   = nn.LayerNorm(d_model)
            self.mlp     = nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Linear(d_model * 4, d_model),
            )
            self.dp2     = DropPath(drop_path_p)

        def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
            # Inject positional encoding into Q and K at every layer
            xn  = self.norm1(x)
            q   = xn + pos
            k   = xn + pos
            v   = xn
            out, _ = self.attn(q, k, v)
            x   = x + self.dp1(out)
            x   = x + self.dp2(self.mlp(self.norm2(x)))
            return x

    # ------------------------------------------------------------------ #
    # Mini PointNet: tokenises one local patch [B·C, 3, k] → [B·C, d_model]
    # ------------------------------------------------------------------ #
    class MiniPointNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp1 = nn.Sequential(
                nn.Conv1d(in_channels, 64, 1, bias=False),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                nn.Conv1d(64, 64, 1, bias=False),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
            )
            self.mlp2 = nn.Sequential(
                nn.Conv1d(128, 128, 1, bias=False),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Conv1d(128, d_model, 1, bias=False),
                nn.BatchNorm1d(d_model),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            h = self.mlp1(x)                                    # [B·C, 64, k]
            g = h.max(dim=2, keepdim=True).values.expand_as(h)
            h = torch.cat([h, g], dim=1)                        # [B·C, 128, k]
            h = self.mlp2(h)                                    # [B·C, d_model, k]
            return h.max(dim=2).values                          # [B·C, d_model]

    # ------------------------------------------------------------------ #
    # Full encoder
    # ------------------------------------------------------------------ #
    class PatchEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.out_dim     = out_dim
            self.n_centers   = n_centers
            self.k_neighbors = k_neighbors
            self.d_model     = d_model

            self.mini_pnet   = MiniPointNet()

            # GELU positional encoding (as in Point-JEPA)
            self.pos_embed   = nn.Sequential(
                nn.Linear(3, 128),
                nn.GELU(),
                nn.Linear(128, d_model),
            )

            self.blocks      = nn.ModuleList([
                TransformerBlock(dpr[i]) for i in range(n_layers)
            ])
            self.norm        = nn.LayerNorm(d_model)

            # Global feature is max‖mean concat → 2·d_model; project to out_dim
            self.proj        = nn.Linear(2 * d_model, out_dim)

        def tokenize(self, xyz):
            """Extract patches and tokenize: xyz [B, N, 3] → tokens [B, C, D], centers [B, C, 3]."""
            B        = xyz.shape[0]
            C, k     = self.n_centers, self.k_neighbors
            ctr_idx  = self._fps(xyz, C)
            centers  = xyz[torch.arange(B, device=xyz.device).unsqueeze(1), ctr_idx]
            nn_idx   = self._knn(xyz, centers, k)
            idx_exp  = nn_idx.unsqueeze(-1).expand(-1, -1, -1, 3)
            patches  = xyz.unsqueeze(1).expand(-1, C, -1, -1).gather(2, idx_exp)
            patches  = patches - centers.unsqueeze(2)           # local coords

            pts    = patches.view(B * C, k, 3).transpose(1, 2)  # [B·C, 3, k]
            tokens = self.mini_pnet(pts).view(B, C, self.d_model)  # [B, C, d_model]
            return tokens, centers

        @staticmethod
        def _fps(xyz, n):
            """Farthest-point sampling: xyz [B, N, 3] → idx [B, n]."""
            B, N, _  = xyz.shape
            device   = xyz.device
            idx      = torch.zeros(B, n, dtype=torch.long, device=device)
            dist     = torch.full((B, N), float("inf"), device=device)
            farthest = torch.randint(0, N, (B,), device=device)
            for i in range(n):
                idx[:, i] = farthest
                c    = xyz[torch.arange(B, device=device), farthest].unsqueeze(1)
                d    = ((xyz - c) ** 2).sum(-1)
                dist = torch.minimum(dist, d)
                farthest = dist.argmax(dim=1)
            return idx

        @staticmethod
        def _knn(xyz, centers, k):
            """xyz [B,N,3], centers [B,C,3] → indices [B,C,k]."""
            d = ((xyz.unsqueeze(1) - centers.unsqueeze(2)) ** 2).sum(-1)
            return d.topk(k, dim=2, largest=False).indices

        def _extract_patches(self, xyz):
            """xyz [B, N, 3] → patches [B, C, k, 3], centers [B, C, 3]."""
            B        = xyz.shape[0]
            C, k     = self.n_centers, self.k_neighbors
            ctr_idx  = self._fps(xyz, C)
            centers  = xyz[torch.arange(B, device=xyz.device).unsqueeze(1), ctr_idx]
            nn_idx   = self._knn(xyz, centers, k)
            idx_exp  = nn_idx.unsqueeze(-1).expand(-1, -1, -1, 3)
            patches  = xyz.unsqueeze(1).expand(-1, C, -1, -1).gather(2, idx_exp)
            patches  = patches - centers.unsqueeze(2)           # local coords
            return patches, centers

        def represent(self, x):
            """x [B, 3, N] → [B, out_dim]. Encodes all tokens (not masked)."""
            xyz = x.transpose(1, 2) if x.shape[1] == in_channels else x  # [B, N, 3]
            tokens, centers = self.tokenize(xyz)                 # [B, C, d_model], [B, C, 3]
            pos = self.pos_embed(centers)                        # [B, C, d_model]

            # Transformer with positional encoding injected at every layer
            for block in self.blocks:
                tokens = block(tokens, pos)
            tokens = self.norm(tokens)                           # [B, C, d_model]

            # Global feature: max + mean (following Point-JEPA svm_validation)
            global_feat = torch.cat(
                [tokens.max(dim=1).values, tokens.mean(dim=1)], dim=-1
            )                                                    # [B, 2·d_model]
            return self.proj(global_feat)                        # [B, out_dim]

        def forward(self, x):
            return self.represent(x)

    return PatchEncoder()

# --------------------------------------------------------------------------- #
# 2) SSL OBJECTIVE  — Point-JEPA
# --------------------------------------------------------------------------- #
def build_ssl(encoder, cfg):
    """Multi-view Point-JEPA: masked predictive learning with rotation invariance.

    Expects two augmented views of the same point cloud with different rotations.
    View1 (student): patches are masked, only context visible
    View2 (teacher): all patches visible, via EMA copy
    Student predicts teacher's encoding of the SAME patches from different rotation,
    forcing learned features to be rotation-invariant.

    Args:
        encoder: PatchEncoder (student backbone)
        cfg: config with predictor_depth, predictor_heads, num_targets, target_ratio, etc.

    Returns:
        PointJEPA module with compute_loss(tokens1, centers1, tokens2, centers2)
    """
    from examples.pointcloud.jepa_modules import EMA, TargetSampler, ContextSampler, Predictor

    predictor_depth = int(getattr(cfg, "predictor_depth", 6))
    predictor_heads = int(getattr(cfg, "predictor_heads", 6))
    num_targets = int(getattr(cfg, "num_targets", 4))
    target_ratio = tuple(getattr(cfg, "target_ratio", (0.15, 0.2)))
    context_ratio = tuple(getattr(cfg, "context_ratio", (0.4, 0.75)))
    loss_beta = float(getattr(cfg, "loss_beta", 2.0))
    ema_tau_min = float(getattr(cfg, "ema_tau_min", 0.99))
    ema_tau_max = float(getattr(cfg, "ema_tau_max", 0.9998))

    class PointJEPA(nn.Module):
        def __init__(self, student_encoder):
            super().__init__()
            self.student = student_encoder
            self.teacher = EMA(student_encoder, tau_min=ema_tau_min, tau_max=ema_tau_max, tau_steps=10000)
            self.predictor = Predictor(
                embed_dim=encoder.d_model,
                depth=predictor_depth,
                num_heads=predictor_heads,
            )
            self.target_sampler = TargetSampler(num_targets=num_targets, target_ratio=target_ratio)
            self.context_sampler = ContextSampler(context_ratio=context_ratio)
            self.pos_embed = nn.Sequential(
                nn.Linear(3, 128),
                nn.GELU(),
                nn.Linear(128, encoder.d_model),
            )
            self.loss_func = nn.SmoothL1Loss(beta=loss_beta)

        def encode_tokens(self, tokens, centers, model):
            """Encode tokenized patches: tokens [B,T,D], centers [B,T,3] -> features [B,T,D]."""
            pos = self.pos_embed(centers)  # [B, T, D]
            for block in model.blocks:
                tokens = block(tokens, pos)
            tokens = model.norm(tokens)
            return tokens

        def compute_loss(self, batch):
            """
            Multi-view JEPA for rotation invariance.

            Args:
                batch: (tokens1, centers1, tokens2, centers2) — two views
                  tokens1 [B,T,D]: view1 (student, will be masked)
                  centers1 [B,T,3]: view1 center positions
                  tokens2 [B,T,D]: view2 (teacher target)
                  centers2 [B,T,3]: view2 center positions

            Returns:
                loss, logs dict
            """
            if isinstance(batch, (tuple, list)) and len(batch) == 4:
                tokens1, centers1, tokens2, centers2 = batch
            else:
                raise ValueError(f"Expected 4-tuple (tokens1, centers1, tokens2, centers2), got {type(batch)}")

            B, T, D = tokens1.shape

            # Sample which patches are targets to be masked in view1
            target_tokens_list, target_indices_list = self.target_sampler(tokens1)

            # Sample context patches (unmasked) in view1
            context_tokens = self.context_sampler(tokens1, target_indices_list)  # [B, n_ctx, D]

            # Encode context (unmasked patches of view1) with student
            context_pos = self.pos_embed(centers1[:, :context_tokens.shape[1]])
            context_features = self.encode_tokens(context_tokens, centers1[:, :context_tokens.shape[1]], self.student)

            # Generate targets from view2 with teacher (no grad)
            # Teacher sees all patches of view2 (different rotation from view1)
            with torch.no_grad():
                teacher_features_v2 = self.encode_tokens(tokens2, centers2, self.teacher.ema_model)

            # Match target indices from view1 to view2 (same patch positions, different rotations)
            loss_total = 0.0
            for m in range(len(target_indices_list)):
                target_idx = target_indices_list[m]  # which patches of view1 were masked

                # Get teacher's encoding of the SAME patches but from view2
                # (since both views are tokenized identically, patch indices align)
                target_feat = teacher_features_v2[:, target_idx]  # [B, n_tgt, D]
                target_pos = self.pos_embed(centers2[:, target_idx])  # [B, n_tgt, D]

                # Predict view2's patches from view1's context
                predicted = self.predictor(context_features, context_pos, target_pos)  # [B, n_tgt, D]

                # Smooth L1 loss: student predicts teacher's rotated view
                loss = self.loss_func(predicted, target_feat.detach())
                loss_total = loss_total + loss

            loss_total = loss_total / max(len(target_indices_list), 1)
            return loss_total, {"loss": loss_total}

        def forward(self, batch):
            return self.compute_loss(batch)

    return PointJEPA(encoder)


@torch.no_grad()
def evaluate_ssl(ssl, loader, device):
    ssl.eval()
    ssl.student.eval()
    ssl.teacher.eval()
    totals = {}
    count = 0
    for batch in loader:
        # batch = (v1, v2, label) from SSL dataset mode
        v1, v2, *_ = batch
        v1 = v1.to(device)
        v2 = v2.to(device)

        # Tokenize both views
        xyz1 = v1.transpose(1, 2)  # [B, N, 3]
        xyz2 = v2.transpose(1, 2)  # [B, N, 3]
        tokens1, centers1 = ssl.student.tokenize(xyz1)
        tokens2, centers2 = ssl.student.tokenize(xyz2)

        loss, logs = ssl.compute_loss((tokens1, centers1, tokens2, centers2))
        count += 1
        totals["loss"] = totals.get("loss", 0.0) + float(loss.item())
        for k, v in logs.items():
            totals[k] = totals.get(k, 0.0) + float(v.item() if torch.is_tensor(v) else v)
    if count == 0:
        return {"loss": 0.0}
    return {k: v / count for k, v in totals.items()}


@torch.no_grad()
def evaluate_probe(encoder, cfg, device):
    from examples.pointcloud.eval import extract_features, probe

    dcfg = PointCloudConfig(**OmegaConf.to_container(cfg.data, resolve=True))
    dcfg_dict = asdict(dcfg)
    n_classes = int(cfg.data.n_classes)

    Xtr, ytr = extract_features(encoder, "train", dcfg_dict, device, mode="supervised")
    Xte, yte = extract_features(encoder, "test", dcfg_dict, device, mode="supervised")
    metrics = probe(Xtr, ytr, Xte, yte, n_classes)

    # Rotated probe: canonical train features vs rotation-augmented test features.
    # Exposes whether the model generalises across the training rotation setting.
    if cfg.data.rotate != "none":
        Xte_rot, yte_rot = extract_features(encoder, "test", dcfg_dict, device, mode="ssl")
        metrics_rot = probe(Xtr, ytr, Xte_rot, yte_rot, n_classes)
        metrics["rotated_accuracy"] = metrics_rot["accuracy"]
        metrics["rotated_gap"] = metrics_rot["gap"]

    return metrics


# --------------------------------------------------------------------------- #
# TRAINING LOOP  — provided
# --------------------------------------------------------------------------- #
def run(fname="examples/pointcloud/cfgs/train.yaml", cfg=None, folder=None, **overrides):
    if cfg is None:
        cfg = OmegaConf.load(fname)
        if overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist([f"{k}={v}" for k, v in overrides.items()]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.meta.seed)

    dcfg = PointCloudConfig(**OmegaConf.to_container(cfg.data, resolve=True))
    dcfg.split = "train"
    dcfg.mode = "ssl"
    loader = make_loader(dcfg)

    dcfg_eval = PointCloudConfig(**OmegaConf.to_container(cfg.data, resolve=True))
    dcfg_eval.split = "test"
    dcfg_eval.mode = "ssl"
    eval_loader = make_loader(dcfg_eval, shuffle=False)

    encoder = build_encoder(cfg.model).to(device)
    ssl = build_ssl(encoder, cfg.model).to(device)
    opt = torch.optim.AdamW(ssl.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)

    wandb_run = setup_wandb(
        project="eb_jepa",
        config={"example": "pointcloud", **OmegaConf.to_container(cfg, resolve=True)},
        run_dir=folder or cfg.meta.ckpt_dir,
        run_name=f"pointcloud_{cfg.data.rotate}",
        tags=["pointcloud", f"rotate_{cfg.data.rotate}", f"seed_{cfg.meta.seed}"],
        group=cfg.logging.get("wandb_group"),
        enabled=cfg.logging.get("log_wandb", False),
        sweep_id=cfg.logging.get("wandb_sweep_id"),
    )

    ckpt_dir = folder or cfg.meta.ckpt_dir
    os.makedirs(ckpt_dir, exist_ok=True)
    eval_every = int(cfg.logging.get("eval_every", 1))
    for epoch in range(cfg.optim.epochs):
        ssl.train()
        for batch in loader:
            # batch = (v1, v2, label) from SSL dataset mode
            # v1, v2 are TWO INDEPENDENT augmented views of the same point cloud
            v1, v2, *_ = batch
            v1 = v1.to(device)
            v2 = v2.to(device)

            # Multi-view JEPA: v1 (student, masked) predicts v2 (teacher, unmasked)
            # This forces rotation invariance since v1 and v2 have different rotations
            xyz1 = v1.transpose(1, 2)  # [B, N, 3]
            xyz2 = v2.transpose(1, 2)  # [B, N, 3]

            with torch.no_grad():
                tokens1, centers1 = encoder.tokenize(xyz1)
                tokens2, centers2 = encoder.tokenize(xyz2)

            # JEPA forward pass: student encodes masked v1, predicts unmasked v2 via teacher
            opt.zero_grad(set_to_none=True)
            loss, logs = ssl.compute_loss((tokens1, centers1, tokens2, centers2))
            loss.backward()
            opt.step()

            # Update teacher EMA
            ssl.teacher.update()

        eval_logs = None
        probe_logs = None
        if eval_every > 0 and (epoch % eval_every == 0 or epoch == cfg.optim.epochs - 1):
            eval_logs = evaluate_ssl(ssl, eval_loader, device)
            probe_logs = evaluate_probe(encoder, cfg, device)
        if wandb_run:
            import wandb

            log_dict = {"epoch": epoch, **{f"train/{k}": v.item() if torch.is_tensor(v) else v for k, v in logs.items()}, "train/loss": loss.item()}
            if eval_logs is not None:
                log_dict.update({f"eval/{k}": v for k, v in eval_logs.items()})
            if probe_logs is not None:
                log_dict.update({f"probe/{k}": v for k, v in probe_logs.items()})
            wandb.log(log_dict)
        print(f"[pointcloud:{cfg.data.rotate}] epoch {epoch} loss={loss.item():.4f} {logs}", flush=True)
        if eval_logs is not None:
            print(f"[pointcloud:{cfg.data.rotate}] epoch {epoch} eval={eval_logs}", flush=True)
        if probe_logs is not None:
            print(f"[pointcloud:{cfg.data.rotate}] epoch {epoch} probe={probe_logs}", flush=True)
        torch.save({"epoch": epoch, "encoder": encoder.state_dict(),
                    "cfg": OmegaConf.to_container(cfg, resolve=True)},
                   os.path.join(ckpt_dir, "latest.pth.tar"))


    if wandb_run:
        import wandb

        wandb.finish()
    print(f"[pointcloud] done -> {ckpt_dir}/latest.pth.tar")


if __name__ == "__main__":
    fname = sys.argv[sys.argv.index("--fname") + 1] if "--fname" in sys.argv \
        else "examples/pointcloud/cfgs/train.yaml"
    run(fname=fname)
