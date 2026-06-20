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
    """TODO: return a PointNet encoder mapping a point cloud [B, 3, N] to a global
    representation [B, D]. Expose `.represent(x) -> [B, D]` (the frozen-feature API
    eval.py calls) and an `.out_dim` attribute.

    Hints: a shared per-point MLP of 1x1 Conv1d layers (3 -> 64 -> 64 -> 128 ->
    out_dim, each Conv1d + BatchNorm1d + ReLU) followed by a symmetric max-pool
    over the N points gives a PERMUTATION-INVARIANT global feature (PointNet, Qi
    et al. 2017; no T-Net needed). The max-pool is what makes it order-agnostic;
    rotation invariance, in contrast, has to be LEARNED from the augmented views."""
    in_channels = int(getattr(cfg, "in_channels", 3))
    out_dim = int(getattr(cfg, "out_dim", 1024))

    class PointNetEncoder(nn.Module):
        def __init__(self, in_ch, out_d):
            super().__init__()
            self.out_dim = out_d
            self.net = nn.Sequential(
                nn.Conv1d(in_ch, 64, kernel_size=1, bias=False),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                nn.Conv1d(64, 64, kernel_size=1, bias=False),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                nn.Conv1d(64, 128, kernel_size=1, bias=False),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Conv1d(128, out_d, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_d),
                nn.ReLU(inplace=True),
            )

        def represent(self, x):
            if x.dim() != 3:
                raise ValueError(f"expected [B, C, N] point cloud tensor, got {tuple(x.shape)}")
            if x.shape[1] != in_channels and x.shape[2] == in_channels:
                x = x.transpose(1, 2)
            h = self.net(x)
            return torch.max(h, dim=2).values

        def forward(self, x):
            return self.represent(x)

    return PointNetEncoder(in_channels, out_dim)


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
    from examples.pointcloud.eval import extract_features, probe, build_random_encoder

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
        print(f"[pointcloud:{cfg.data.rotate}] epoch {epoch} loss={loss.item():.4f} {logs}", flush=True)
        if eval_logs is not None:
            print(f"[pointcloud:{cfg.data.rotate}] epoch {epoch} eval={eval_logs}", flush=True)
        if eval_every > 0 and (epoch % eval_every == 0 or epoch == cfg.optim.epochs - 1):
            print(f"[pointcloud:{cfg.data.rotate}] epoch {epoch} probe={probe_logs}", flush=True)
        torch.save({"epoch": epoch, "encoder": encoder.state_dict(),
                    "cfg": OmegaConf.to_container(cfg, resolve=True)},
                   os.path.join(ckpt_dir, "latest.pth.tar"))
        for batch in loader:
            batch = batch.to(device) if torch.is_tensor(batch) else [b.to(device) for b in batch]
            opt.zero_grad(set_to_none=True)
            loss, logs = ssl.compute_loss(batch)
            loss.backward(); opt.step()
        eval_logs = None
        if eval_every > 0 and (epoch % eval_every == 0 or epoch == cfg.optim.epochs - 1):
            eval_logs = evaluate_ssl(ssl, eval_loader, device)
            probe_logs = evaluate_probe(encoder, cfg, device)
        if wandb_run:
            import wandb

            log_dict = {"epoch": epoch, **{f"train/{k}": v.item() if torch.is_tensor(v) else v for k, v in logs.items()}, "train/loss": loss.item()}
            if eval_logs is not None:
                log_dict.update({f"eval/{k}": v for k, v in eval_logs.items()})
            if eval_every > 0 and (epoch % eval_every == 0 or epoch == cfg.optim.epochs - 1):
                log_dict.update({f"probe/{k}": v for k, v in probe_logs.items()})
            wandb.log(log_dict)
        

    if wandb_run:
        import wandb

        wandb.finish()
    print(f"[pointcloud] done -> {ckpt_dir}/latest.pth.tar")


if __name__ == "__main__":
    fname = sys.argv[sys.argv.index("--fname") + 1] if "--fname" in sys.argv \
        else "examples/pointcloud/cfgs/train.yaml"
    run(fname=fname)
