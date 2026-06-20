"""PointCloud — SSL pretraining entrypoint (view-invariant 3D shape SSL).

Research question: can a two-view SSL objective learn a VIEW-INVARIANT shape
representation on an unordered/irregular modality (point clouds), and how does the
linear-probe accuracy degrade as we demand more rotation invariance (none -> z ->
SO(3))?

Point clouds have no temporal frames, so the objective is a two-view VICReg (the
image-JEPA / audio / EEG recipe), NOT a predictive JEPA. Two independent augmented
samplings + rotations of the same object are the two views.

The DATA + TRAINING LOOP are provided. The two modelling pieces you implement are
marked `# TODO` below — that is the whole point of the track:
  1. the PointNet encoder over [B, 3, N]
  2. the two-view VICReg objective

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
    in_channels = int(getattr(cfg, "in_channels", 3))
    out_dim     = int(getattr(cfg, "out_dim", 1024))
    n_centers   = int(getattr(cfg, "n_centers", 64))
    k_neighbors = int(getattr(cfg, "k_neighbors", 32))
    d_model     = int(getattr(cfg, "d_model", 384))
    n_heads     = int(getattr(cfg, "n_heads", 6))
    n_layers    = int(getattr(cfg, "n_layers", 12))

    class MiniPointNet(nn.Module):
        """Tokenise un patch [B·C, 3, k] → patch embedding [B·C, d_model]."""
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
            # x : [B·C, 3, k]
            h = self.mlp1(x)                        # [B·C, 64, k]
            g = h.max(dim=2, keepdim=True).values.expand_as(h)
            h = torch.cat([h, g], dim=1)            # [B·C, 128, k]
            h = self.mlp2(h)                        # [B·C, d_model, k]
            return h.max(dim=2).values              # [B·C, d_model]

    class PatchEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.out_dim     = out_dim
            self.n_centers   = n_centers
            self.k_neighbors = k_neighbors
            self.d_model     = d_model

            # Tokenisation
            self.mini_pnet = MiniPointNet()

            # Encodage positionnel (depuis les centres 3D → d_model)
            self.pos_embed = nn.Sequential(
                nn.Linear(3, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, d_model),
            )

            # Transformer encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=0.0,
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

            # Projection finale vers out_dim si d_model != out_dim
            self.proj = nn.Linear(d_model, out_dim) if d_model != out_dim else nn.Identity()

        @staticmethod
        def _fps(xyz, n):
            """xyz : [B, N, 3] → idx : [B, n]"""
            B, N, _ = xyz.shape
            device  = xyz.device
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
            """xyz : [B,N,3], centers : [B,C,3] → [B,C,k]"""
            d = ((xyz.unsqueeze(1) - centers.unsqueeze(2)) ** 2).sum(-1)  # [B,C,N]
            return d.topk(k, dim=2, largest=False).indices                  # [B,C,k]

        def _extract_patches(self, xyz):
            """xyz : [B, N, 3] → patches [B, C, k, 3], centres [B, C, 3]"""
            B = xyz.shape[0]
            C, k = self.n_centers, self.k_neighbors
            ctr_idx = self._fps(xyz, C)                                      # [B, C]
            centers = xyz[torch.arange(B, device=xyz.device).unsqueeze(1),
                          ctr_idx]                                            # [B, C, 3]
            nn_idx  = self._knn(xyz, centers, k)                             # [B, C, k]
            idx_exp = nn_idx.unsqueeze(-1).expand(-1, -1, -1, 3)
            patches = xyz.unsqueeze(1).expand(-1, C, -1, -1).gather(2, idx_exp)
            patches = patches - centers.unsqueeze(2)                         # normalisation locale
            return patches, centers

        def represent(self, x):
            """x : [B, 3, N] → [B, out_dim]"""
            if x.shape[1] == in_channels:
                xyz = x.transpose(1, 2)   # [B, N, 3]
            else:
                xyz = x

            B = xyz.shape[0]
            C, k = self.n_centers, self.k_neighbors

            patches, centers = self._extract_patches(xyz)  # [B, C, k, 3], [B, C, 3]

            # Tokenisation via mini PointNet
            pts = patches.view(B * C, k, 3).transpose(1, 2)  # [B·C, 3, k]
            tokens = self.mini_pnet(pts)                       # [B·C, d_model]
            tokens = tokens.view(B, C, self.d_model)          # [B, C, d_model]

            # Encodage positionnel des centres
            pos = self.pos_embed(centers)                      # [B, C, d_model]
            tokens = tokens + pos

            # Transformer : interaction entre patches
            tokens = self.transformer(tokens)                  # [B, C, d_model]

            # Agrégation globale
            global_feat = tokens.max(dim=1).values             # [B, d_model]

            return self.proj(global_feat)                      # [B, out_dim]

        def forward(self, x):
            return self.represent(x)

    return PatchEncoder()

# --------------------------------------------------------------------------- #
# 2) SSL OBJECTIVE  — # TODO
# --------------------------------------------------------------------------- #
def build_ssl(encoder, cfg):
    """TODO: return an nn.Module exposing `compute_loss(batch) -> (loss, logs)`,
    where `batch = (v1, v2, label)` are the two augmented views (label unused for
    SSL).

    Build a two-view VICReg head:
      v1, v2 -> encoder.represent -> eb_jepa.architectures.Projector ->
      eb_jepa.losses.VICRegLoss(std_coeff, cov_coeff) on the two projections.
    The variance + covariance terms are the anti-collapse ingredient; the
    invariance (MSE) term is what pulls the two views of the same object together
    and makes the representation VIEW-INVARIANT. Return the scalar loss and a logs
    dict (e.g. the VICRegLoss component breakdown)."""
    projector_spec = getattr(cfg, "projector", None) or getattr(cfg, "proj", None)
    if projector_spec is None:
        projector_spec = f"{encoder.out_dim}-2048-2048"
    elif isinstance(projector_spec, (list, tuple)):
        projector_spec = "-".join(str(int(v)) for v in projector_spec)
        if not projector_spec.startswith(f"{encoder.out_dim}-"):
            projector_spec = f"{encoder.out_dim}-{projector_spec}"
    else:
        projector_spec = str(projector_spec)
        if not projector_spec.startswith(f"{encoder.out_dim}-"):
            projector_spec = f"{encoder.out_dim}-{projector_spec}"

    std_coeff = float(getattr(cfg, "std_coeff", 25.0))
    cov_coeff = float(getattr(cfg, "cov_coeff", 1.0))

    class TwoViewVICReg(nn.Module):
        def __init__(self, enc):
            super().__init__()
            from eb_jepa.architectures import Projector
            from eb_jepa.losses import VICRegLoss

            self.encoder = enc
            self.projector = Projector(projector_spec)
            self.criterion = VICRegLoss(std_coeff=std_coeff, cov_coeff=cov_coeff)

        def compute_loss(self, batch):
            v1, v2, *_ = batch
            z1 = self.projector(self.encoder.represent(v1))
            z2 = self.projector(self.encoder.represent(v2))
            loss_dict = self.criterion(z1, z2)
            return loss_dict["loss"], loss_dict

    return TwoViewVICReg(encoder)


@torch.no_grad()
def evaluate_ssl(ssl, loader, device):
    ssl.eval()
    totals = {}
    count = 0
    for batch in loader:
        batch = batch.to(device) if torch.is_tensor(batch) else [b.to(device) for b in batch]
        loss, logs = ssl.compute_loss(batch)
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
            batch = batch.to(device) if torch.is_tensor(batch) else [b.to(device) for b in batch]
            opt.zero_grad(set_to_none=True)
            loss, logs = ssl.compute_loss(batch)
            loss.backward(); opt.step()
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
