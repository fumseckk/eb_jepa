"""t-SNE visualization of exported latent representations.

Usage:
    uv run python -m examples.pointcloud.tsne_viz \\
        --npz path/to/latents_test.npz \\
        --out tsne.png          # optional, defaults to <npz_dir>/tsne.png
        --perplexity 40         # optional
        --pca-dim 50            # optional, 0 to skip PCA pre-reduction
        --seed 0                # optional
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

MODELNET40_CLASSES = [
    "airplane", "bathtub", "bed", "bench", "bookshelf", "bottle", "bowl", "car",
    "chair", "cone", "cup", "curtain", "desk", "door", "dresser", "flower_pot",
    "glass_box", "guitar", "keyboard", "lamp", "laptop", "mantel", "monitor",
    "night_stand", "person", "piano", "plant", "radio", "range_hood", "sink",
    "sofa", "stairs", "stool", "table", "tent", "toilet", "tv_stand", "vase",
    "wardrobe", "xbox",
]


def build_colormap(n):
    """Return a list of n visually distinct RGBA colors."""
    tab20b = plt.cm.tab20b.colors   # 20 colors
    tab20c = plt.cm.tab20c.colors   # 20 colors
    palette = list(tab20b) + list(tab20c)
    return [palette[i % len(palette)] for i in range(n)]


def run(npz_path: str, out_path: str | None, perplexity: int, pca_dim: int, seed: int):
    data = np.load(npz_path)
    X = data["features"].astype(np.float32)   # [N, D]
    y = data["labels"].astype(np.int64)        # [N]

    n_classes = int(y.max()) + 1
    class_names = MODELNET40_CLASSES[:n_classes] if n_classes <= len(MODELNET40_CLASSES) \
        else [str(i) for i in range(n_classes)]

    print(f"Loaded {X.shape[0]} samples, dim={X.shape[1]}, {n_classes} classes", flush=True)

    if pca_dim > 0 and X.shape[1] > pca_dim:
        print(f"PCA {X.shape[1]} → {pca_dim} dims …", flush=True)
        X = PCA(n_components=pca_dim, random_state=seed).fit_transform(X)

    print(f"t-SNE (perplexity={perplexity}) …", flush=True)
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        max_iter=1000,
        random_state=seed,
        n_jobs=-1,
    )
    Z = tsne.fit_transform(X)   # [N, 2]

    # ------------------------------------------------------------------ #
    # Plot
    # ------------------------------------------------------------------ #
    colors = build_colormap(n_classes)
    fig, ax = plt.subplots(figsize=(14, 11))

    for c in range(n_classes):
        mask = y == c
        if mask.sum() == 0:
            continue
        ax.scatter(
            Z[mask, 0], Z[mask, 1],
            c=[colors[c]],
            s=20,
            alpha=0.7,
            linewidths=0,
            label=None,
            # label=class_names[c],
        )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("t-SNE of latent representations (test set)", fontsize=14)

    # Legend: two columns to keep it compact
    # handles = [
    #     mpatches.Patch(color=colors[c], label=class_names[c])
    #     for c in range(n_classes)
    # ]
    # ax.legend(
    #     handles=handles,
    #     fontsize=7,
    #     ncol=2,
    #     loc="upper left",
    #     bbox_to_anchor=(1.01, 1),
    #     borderaxespad=0,
    #     framealpha=0.8,
    # )

    fig.tight_layout()

    out = Path(out_path) if out_path else Path(npz_path).with_name("tsne.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True, help="Path to latents_test.npz")
    parser.add_argument("--out", default=None, help="Output PNG path (default: <npz_dir>/tsne.png)")
    parser.add_argument("--perplexity", type=int, default=40)
    parser.add_argument("--pca-dim", type=int, default=50, dest="pca_dim",
                        help="PCA pre-reduction dim (0 = skip)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run(args.npz, args.out, args.perplexity, args.pca_dim, args.seed)


if __name__ == "__main__":
    main()
