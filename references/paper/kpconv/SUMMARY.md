# KPConv — Flexible and Deformable Convolution for Point Clouds

**Authors:** Hugues Thomas, Charles R. Qi, Jean-Emmanuel Deschaud, Beatriz
Marcotegui, François Goulette, Leonidas J. Guibas **Venue/Year:** ICCV 2019
**PDF:**
https://geometry.stanford.edu/lgl_2024/papers/tqdmgg-KPconv-iccv19/tqdmgg-KPconv-iccv19.pdf
**Repo:** https://github.com/HuguesTHOMAS/KPConv (MIT)

## TL;DR

Defines a true **convolution** directly on irregular point clouds — no
voxelization, no graph. A kernel is a set of **K learnable 3D "kernel points"**,
each carrying a weight matrix; the contribution of a neighbour is weighted by a
(linear) correlation between the neighbour and each kernel point. Comes in
**rigid** and **deformable** variants. Long the SOTA on S3DIS / Scannet /
ModelNet40, and still the strongest geometrically-grounded point-conv primitive.

## Mathematical formulation

Input: points $X=\{x_i \in \mathbb{R}^d\}$, features
$F^{in}\in\mathbb{R}^{n\times c_{in}}$. For a query $q$, radius neighbourhood
$\mathcal{N}(q)=\{i:\|x_i-q\|\le r\}$. Kernel =
$\{\tilde{x}_k\in\mathbb{R}^d\}_{k=1}^K$ with weights
$W_k\in\mathbb{R}^{c_{in}\times c_{out}}$. Linear (tent) correlation:
$$h(y_i,\tilde{x}_k)=\max\!\left(0,\;1-\frac{\|y_i-\tilde{x}_k\|}{\sigma}\right),\quad y_i=x_i-q$$
Kernel function: $$g(y_i)=\sum_{k<K} h(y_i,\tilde{x}_k)\,W_k$$ Convolution at
$q$: $$y(q)=\sum_{x_i\in\mathcal{N}(q)} g(x_i-q)\,f_i$$ Deformable variant:
kernel points $\tilde{x}_k$ are themselves predicted from the local feature.

## Reported numbers (official repo)

- ModelNet40 OA: rigid 92.9 %, deform 92.7 %.
- ShapeNetPart inst. mIoU: rigid 86.2 %, deform 86.4 %.
- S3DIS mIoU: rigid 65.4 %, deform **67.1 %**.
- Scannet mIoU: 68.6 %; Semantic3D 74.6 %; NPM3D 82.0 % (deform).

## Relevance to the EB-JEPA hackathon — using KPConv _as the JEPA encoder_

The current `examples/pointcloud/main.py` `# TODO: build_encoder` specifies a
plain PointNet (shared MLP + max-pool). KPConv is a drop-in upgrade that is
**geometrically far stronger**:

1. **Why KPConv over PointNet/Transformer here.** JEPA predicts in latent space,
   so the encoder must build geometry-aware features — exactly KPConv's
   strength. PointNet's max-pool is permutation-invariant but discards local
   structure; Transformers (Point-JEPA) need an external tokenizer. KPConv gives
   local geometric aggregation for free.
2. **Encoder recipe.** Replace the PointNet stack with a small KPConv encoder
   (e.g. 4–5 ResKPConv blocks, rigid kernels, U-Net downsampling), producing a
   per-point or global feature. Expose `.represent()` and `.out_dim` exactly as
   the README requires. The HuguesTHOMAS/KPConv repo is MIT-licensed and
   PyTorch-based — the `models/KPConv.py` backbone can be lifted directly.
3. **Context/target on point clouds.** Two viable recipes, both compatible with
   KPConv:
   - **Two-view VICReg** (current `examples/pointcloud` recipe): two augmented
     samplings of the same object → KPConv encoder → Projector → VICRegLoss.
     KPConv's geometric bias should tighten the view-invariance study (none → z
     → SO(3)) compared to PointNet.
   - **Predictive JEPA** (Point-JEPA style): KPConv encodes the _context_
     points; an EMA KPConv encodes _target_ patches; a predictor predicts target
     embeddings from context + target positions. KPConv's dense per-point
     features make the target/context split natural — mask spatial regions in
     the radius neighbourhood rather than patch indices.
4. **Deformable KPConv + JEPA.** The deformable variant learns kernel-point
   offsets from features, which is itself a latent prediction — an interesting
   alignment with JEPA's "predict in latent space" philosophy, and a candidate
   ablation.

## Caveats / open threads

- Native KPConv repo has custom CUDA ops (`tf_custom_ops` / `cpp_wrappers`);
  PyTorch CUDA build is needed. The community `torch_geometric.nn.KPConv` port
  is easier but slightly slower / less feature-complete.
- ~14.2 M params for the full model — heavier than the PointNet baseline the
  track asks for.
- Original code is TF1-era; prefer the PyTorch KPConvX line (see
  `paper/kpconvx/`) for modern training infra.
