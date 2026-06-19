"""PointCloud — downstream evaluation (answers the view-invariance question).

The feature-extraction harness is provided. What you implement (`# TODO`) is the
linear probe + metric on the official ModelNet40 test split, and the comparison
that makes the result meaningful: the frozen SSL encoder vs a random-encoder floor
(and ideally the same probe across rotate=none|z|so3 to expose the invariance gap).

Run:  python -m examples.pointcloud.eval --ckpt <.../latest.pth.tar>
"""
import sys

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from eb_jepa.datasets.pointcloud.dataset import PointCloudConfig, PointCloudDataset
from examples.pointcloud.main import build_encoder


@torch.no_grad()
def extract_features(encoder, split, dcfg, device):
    """Provided: frozen encoder -> [N, D] features + labels for `split`.

    Uses the deterministic clean (supervised-mode) view so the probe sees one
    canonical sampling per shape."""
    cfg = PointCloudConfig(**{**dcfg, "split": split, "mode": "supervised"})
    ds = PointCloudDataset(cfg)
    loader = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=False, num_workers=8)
    X, y = [], []
    for xb, yb in loader:
        X.append(encoder.represent(xb.to(device)).cpu().numpy())
        y.append(np.asarray(yb))
    return np.concatenate(X), np.concatenate(y)


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


def main():
    ckpt = sys.argv[sys.argv.index("--ckpt") + 1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = OmegaConf.create(state["cfg"])
    encoder = build_encoder(cfg.model).to(device)
    encoder.load_state_dict(state["encoder"]); encoder.eval()

    dcfg = OmegaConf.to_container(cfg.data, resolve=True)
    Xtr, ytr = extract_features(encoder, "train", dcfg, device)
    Xte, yte = extract_features(encoder, "test", dcfg, device)
    print("[pointcloud-eval]", probe(Xtr, ytr, Xte, yte, dcfg["n_classes"]))


if __name__ == "__main__":
    main()
