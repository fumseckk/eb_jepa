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
from examples.pointcloud.main import build_encoder
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
def probe(Xtr, ytr, Xte, yte, n_classes, return_predictions=False):
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
        preds = clf.predict(Xte)
        acc = float((preds == yte).mean() * 100.0)
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
            preds = preds.cpu().numpy()

    metrics = {"accuracy": acc, "chance": chance, "gap": acc - chance}
    if return_predictions:
        return metrics, np.asarray(preds, dtype=np.int64)
    return metrics


def build_per_class_rows(y_true, y_pred, n_classes):
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.int64).reshape(-1)

    rows = []
    for class_id in range(int(n_classes)):
        mask = y_true == class_id
        total = int(mask.sum())
        correct = int((y_pred[mask] == class_id).sum()) if total > 0 else 0
        accuracy = (100.0 * correct / total) if total > 0 else float("nan")
        rows.append(
            {
                "class_id": class_id,
                "accuracy": float(accuracy),
                "correct": correct,
                "total": total,
            }
        )
    return rows


def export_latents(path, latents_dict):
        """Export latent embeddings + labels for downstream analysis.

        Saves a compressed NPZ file with arrays such as:
            - X_train, y_train
            - X_test, y_test
            - X_test_rotated, y_test_rotated (when available)
        """
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, **latents_dict)
        return out_path


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
    metrics, preds = probe(Xtr, ytr, Xte, yte, n_classes, return_predictions=True)
    
    # Compute random encoder baseline for sanity check
    random_encoder = build_random_encoder(Xtr.shape[1], device)
    Xtr_random, _ = extract_features(random_encoder, "train", dcfg, device)
    Xte_random, _ = extract_features(random_encoder, "test", dcfg, device)
    metrics_random, preds_random = probe(
        Xtr_random, ytr, Xte_random, yte, n_classes, return_predictions=True
    )

    per_class_rows = build_per_class_rows(yte, preds, n_classes)
    per_class_rows_random = build_per_class_rows(yte, preds_random, n_classes)
    
    # Rotated probe: train features are canonical, test features use the training
    # rotation augmentation. This reveals whether the model is actually rotation-invariant.
    rotate = dcfg.get("rotate", "none")
    Xte_rot = None
    yte_rot = None
    if rotate != "none":
        Xte_rot, yte_rot = extract_features(encoder, "test", dcfg, device, mode="ssl")
        metrics_rot = probe(Xtr, ytr, Xte_rot, yte_rot, n_classes)
        metrics["rotated_accuracy"] = metrics_rot["accuracy"]
        metrics["rotated_gap"] = metrics_rot["gap"]

    # Merge metrics with random baseline prefixed
    metrics_with_baseline = {**metrics, **{f"random_{k}": v for k, v in metrics_random.items()}}

    # Export latent embeddings for downstream data analysis
    output_dir = Path(folder) if folder is not None else Path(ckpt).parent
    rotate_name = str(rotate)
    latents_path = output_dir / f"latents_{rotate_name}.npz"
    latents_payload = {
        "X_train": np.asarray(Xtr, dtype=np.float32),
        "y_train": np.asarray(ytr, dtype=np.int64),
        "X_test": np.asarray(Xte, dtype=np.float32),
        "y_test": np.asarray(yte, dtype=np.int64),
        "X_train_random": np.asarray(Xtr_random, dtype=np.float32),
        "X_test_random": np.asarray(Xte_random, dtype=np.float32),
    }
    if Xte_rot is not None and yte_rot is not None:
        latents_payload["X_test_rotated"] = np.asarray(Xte_rot, dtype=np.float32)
        latents_payload["y_test_rotated"] = np.asarray(yte_rot, dtype=np.int64)
    latents_path = export_latents(latents_path, latents_payload)
    metrics_with_baseline["latents_path"] = str(latents_path)

    own_wandb_run = None
    if wandb_run is None:
        own_wandb_run = setup_wandb(
            project="eb_jepa",
            config={"example": "pointcloud_eval", **OmegaConf.to_container(cfg, resolve=True)},
            run_dir=folder or str(Path(ckpt).parent),
            run_name=f"pointcloud_eval_{cfg.get('data', {}).get('rotate', dcfg.get('rotate', 'none'))}",
            tags=["pointcloud", "pointcloud_eval"],
            group=cfg.get("logging", {}).get("wandb_group") if hasattr(cfg, "get") else None,
            enabled=cfg.get("logging", {}).get("log_wandb", False) if hasattr(cfg, "get") else False,
            sweep_id=cfg.get("logging", {}).get("wandb_sweep_id") if hasattr(cfg, "get") else None,
        )
        wandb_run = own_wandb_run

    if wandb_run:
        import wandb

        per_class_table = wandb.Table(
            columns=[
                "class_id",
                "accuracy",
                "random_accuracy",
                "correct",
                "total",
            ]
        )
        for row, row_random in zip(per_class_rows, per_class_rows_random):
            per_class_table.add_data(
                row["class_id"],
                row["accuracy"],
                row_random["accuracy"],
                row["correct"],
                row["total"],
            )

        wandb.log(
            {
                **{f"eval/{k}": v for k, v in metrics_with_baseline.items()},
                "eval/per_class_accuracy": per_class_table,
            }
        )
        if hasattr(wandb, "save"):
            wandb.save(str(latents_path), base_path=str(latents_path.parent))
        if own_wandb_run is not None:
            wandb.finish()

    print("[pointcloud-eval]", metrics_with_baseline)
    if folder is not None:
        Path(folder).mkdir(parents=True, exist_ok=True)
        with open(Path(folder) / "metrics.json", "w", encoding="utf-8") as f:
            import json

            json.dump(metrics_with_baseline, f, indent=2, sort_keys=True)
    return metrics_with_baseline


def main():
    ckpt = sys.argv[sys.argv.index("--ckpt") + 1]
    fname = sys.argv[sys.argv.index("--fname") + 1] if "--fname" in sys.argv \
        else "examples/pointcloud/cfgs/eval.yaml"
    cfg = OmegaConf.load(fname)
    cfg.ckpt = ckpt
    run(fname=fname, cfg=cfg)


if __name__ == "__main__":
    main()
