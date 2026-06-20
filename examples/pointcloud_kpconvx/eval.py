"""PointCloud — downstream evaluation (answers the view-invariance question).

The feature-extraction harness is provided. What you implement (`# TODO`) is the
linear probe + metric on the official ModelNet40 test split, and the comparison
that makes the result meaningful: the frozen SSL encoder vs a random-encoder floor
(and ideally the same probe across rotate=none|z|so3 to expose the invariance gap).

Run:  python -m examples.pointcloud.eval --ckpt <.../latest.pth.tar>
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from eb_jepa.datasets.pointcloud.dataset import PointCloudConfig, PointCloudDataset
from examples.pointcloud_kpconvx.main import build_encoder
from eb_jepa.training_utils import setup_wandb


def build_random_encoder(out_dim, device):
    """Build an untrained (random weight) encoder for baseline comparison.
    
    To avoid accidental leakage where an untrained architecture might
    inadvertently correlate with labels (e.g. due to BN / running stats),
    return a pure random-feature encoder that emits iid Gaussian features
    independent of the input. The downstream probe on these features should
    be at chance level.

    The returned object exposes the same `.represent(x)` API and an
    `.out_dim` attribute so it can be used interchangeably with the real
    encoder in the evaluation harness.
    """
    class RandomEncoder(torch.nn.Module):
        def __init__(self, out_d, dev):
            super().__init__()
            self.out_dim = out_d
            self.device = dev

        @torch.no_grad()
        def represent(self, x):
            b = x.shape[0]
            return torch.randn(b, self.out_dim, device=self.device)

    return RandomEncoder(out_dim, device)


@torch.no_grad()
def extract_features(encoder, split, dcfg, device, mode="supervised"):
    """Frozen encoder -> [N, D] features + labels for `split`.

    mode="supervised": deterministic clean (canonical) view, no rotation.
    mode="ssl": one randomly-augmented view using the training rotation setting,
                so the rotated probe can measure true rotation invariance."""
    was_training = encoder.training
    encoder.eval()
    try:
        cfg = PointCloudConfig(**{**dcfg, "split": split, "mode": mode})
        ds = PointCloudDataset(cfg)
        loader = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=False, num_workers=8)
        X, y = [], []
        for batch in loader:
            if mode == "ssl":
                xb, _, yb = batch   # (v1, v2, label) — use first augmented view
            else:
                xb, yb = batch
            X.append(encoder.represent(xb.to(device)).cpu().numpy())
            y.append(np.asarray(yb))
        return np.concatenate(X), np.concatenate(y)
    finally:
        encoder.train(was_training)


# --------------------------------------------------------------------------- #
# PROBE + METRIC  — # TODO
# --------------------------------------------------------------------------- #
def probe(Xtr, ytr, Xte, yte, n_classes):
    """TODO: fit a linear probe on the FROZEN train features (no leakage:
    standardize on train only) and score 40-way shape classification on the
    official test split. Return a metrics dict.
      * accuracy (top-1) on the [N, D] features — sklearn LogisticRegression (or a
        torch nn.Linear trained with cross-entropy) over the frozen features.
      * report it against chance (= 100 / n_classes = 2.5%).
    To make the number meaningful, also run this probe on a RANDOM untrained
    encoder (floor), and ideally compare rotate=none|z|so3 checkpoints — accuracy
    should drop monotonically as more rotation invariance is demanded."""
    Xtr = np.asarray(Xtr, dtype=np.float32)
    Xte = np.asarray(Xte, dtype=np.float32)
    ytr = np.asarray(ytr, dtype=np.int64).reshape(-1)
    yte = np.asarray(yte, dtype=np.int64).reshape(-1)

    mu = Xtr.mean(axis=0, keepdims=True)
    sigma = Xtr.std(axis=0, keepdims=True) + 1e-6
    Xtr = (Xtr - mu) / sigma
    Xte = (Xte - mu) / sigma

    chance = 100.0 / float(n_classes)

    try:
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(
            max_iter=2000,
            multi_class="multinomial",
            solver="lbfgs",
            n_jobs=1,
        )
        clf.fit(Xtr, ytr)
        acc = float(clf.score(Xte, yte) * 100.0)
    except Exception:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        Xtr_t = torch.from_numpy(Xtr).to(device)
        ytr_t = torch.from_numpy(ytr).to(device)
        Xte_t = torch.from_numpy(Xte).to(device)
        yte_t = torch.from_numpy(yte).to(device)

        model = nn.Linear(Xtr.shape[1], n_classes).to(device)
        opt = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=100, line_search_fn="strong_wolfe")
        loss_fn = nn.CrossEntropyLoss()

        def closure():
            opt.zero_grad(set_to_none=True)
            logits = model(Xtr_t)
            loss = loss_fn(logits, ytr_t)
            loss.backward()
            return loss

        opt.step(closure)
        with torch.no_grad():
            preds = model(Xte_t).argmax(dim=1)
            acc = float((preds == yte_t).float().mean().item() * 100.0)

    return {"accuracy": acc, "chance": chance, "gap": acc - chance}


def run(
    fname="examples/pointcloud/cfgs/eval.yaml",
    cfg=None,
    folder=None,
    wandb_run=None,
    **overrides,
):
    if cfg is None:
        cfg = OmegaConf.load(fname)
        if overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist([f"{k}={v}" for k, v in overrides.items()]))

    ckpt = getattr(cfg, "ckpt", None)
    if not ckpt or str(ckpt).startswith("UPDATEME"):
        raise ValueError(
            "PointCloud evaluation needs a checkpoint path. Pass it as --ckpt <.../latest.pth.tar>."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(ckpt, map_location=device, weights_only=False)
    train_cfg = OmegaConf.create(state["cfg"])
    encoder = build_encoder(train_cfg.model).to(device)
    encoder.load_state_dict(state["encoder"])
    encoder.eval()

    eval_data = OmegaConf.create(OmegaConf.to_container(getattr(cfg, "data", {}), resolve=True))
    dcfg = OmegaConf.to_container(OmegaConf.merge(train_cfg.data, eval_data), resolve=True)
    Xtr, ytr = extract_features(encoder, "train", dcfg, device, mode="supervised")
    Xte, yte = extract_features(encoder, "test", dcfg, device, mode="supervised")
    n_classes = int(getattr(cfg, "n_classes", dcfg["n_classes"]))
    metrics = probe(Xtr, ytr, Xte, yte, n_classes)
    
    # Compute random encoder baseline for sanity check
    random_encoder = build_random_encoder(Xtr.shape[1], device)
    Xtr_random, _ = extract_features(random_encoder, "train", dcfg, device)
    Xte_random, _ = extract_features(random_encoder, "test", dcfg, device)
    metrics_random = probe(Xtr_random, ytr, Xte_random, yte, n_classes)
    
    # Merge metrics with random baseline prefixed
    metrics_with_baseline = {**metrics, **{f"random_{k}": v for k, v in metrics_random.items()}}

    # Rotated probe: train features are canonical, test features use the training
    # rotation augmentation. This reveals whether the model is actually rotation-invariant.
    rotate = dcfg.get("rotate", "none")
    if rotate != "none":
        Xte_rot, yte_rot = extract_features(encoder, "test", dcfg, device, mode="ssl")
        metrics_rot = probe(Xtr, ytr, Xte_rot, yte_rot, n_classes)
        metrics["rotated_accuracy"] = metrics_rot["accuracy"]
        metrics["rotated_gap"] = metrics_rot["gap"]

    own_wandb_run = None
    if wandb_run is None:
        own_wandb_run = setup_wandb(
            project="eb_jepa",
            config={"example": "pointcloud_kpconvx_eval", **OmegaConf.to_container(cfg, resolve=True)},
            run_dir=folder or str(Path(ckpt).parent),
            run_name=f"pointcloud_kpconvx_eval_{cfg.get('data', {}).get('rotate', dcfg.get('rotate', 'none'))}",
            tags=["pointcloud", "pointcloud_kpconvx_eval", "kpconvx"],
            group=cfg.get("logging", {}).get("wandb_group") if hasattr(cfg, "get") else None,
            enabled=cfg.get("logging", {}).get("log_wandb", False) if hasattr(cfg, "get") else False,
            sweep_id=cfg.get("logging", {}).get("wandb_sweep_id") if hasattr(cfg, "get") else None,
        )
        wandb_run = own_wandb_run

    if wandb_run:
        import wandb

        wandb.log({f"eval/{k}": v for k, v in metrics_with_baseline.items()})
        if own_wandb_run is not None:
            wandb.finish()

    print("[pointcloud_kpconvx-eval]", metrics_with_baseline)
    if folder is not None:
        Path(folder).mkdir(parents=True, exist_ok=True)
        with open(Path(folder) / "metrics.json", "w", encoding="utf-8") as f:
            import json

            json.dump(metrics_with_baseline, f, indent=2, sort_keys=True)
    return metrics_with_baseline


def main():
    ckpt = sys.argv[sys.argv.index("--ckpt") + 1]
    fname = sys.argv[sys.argv.index("--fname") + 1] if "--fname" in sys.argv \
        else "examples/pointcloud_kpconvx/cfgs/eval.yaml"
    cfg = OmegaConf.load(fname)
    cfg.ckpt = ckpt
    run(fname=fname, cfg=cfg)


if __name__ == "__main__":
    main()
