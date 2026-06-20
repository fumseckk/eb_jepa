# Point-Cloud Decoders — FoldingNet, AtlasNet, TopNet, PCN, SeedFormer

A family of classical-to-modern **decoders that map a latent code
$z \in \mathbb{R}^D$ to a point cloud $\hat{X} \in \mathbb{R}^{N \times 3}$**.
Collected because the user asked specifically about "a decoder that decodes to a
point cloud" for an action-conditioned point-cloud JEPA. These are the candidate
decoder heads when the latent → point-cloud mapping is _not_ the trivial
flow-addition used by PointWorld (see `paper/pointworld/`).

## When you need which

- **Trivial flow decoder** (PointWorld):
  $\hat{x}^{(t+1)} = x^{(t)} + D_\phi(z_{t+1})$ where $D_\phi$ outputs a
  per-point flow. Best when you have the current cloud and only need the delta.
  No generative decoder needed.
- **Generative decoder** (below): map a latent $z$ directly to a full point
  cloud $\hat{X}$. Needed when you want to _imagine_ a cloud from a latent
  without a reference cloud, e.g. for goal-conditioned planning where the goal
  is a latent.

## The decoders

### FoldingNet (Yang et al., CVPR 2018)

- [PDF](https://openaccess.thecvf.com/content_cvpr_2018/papers/Yang_FoldingNet_Point_Cloud_CVPR_2018_paper.pdf)
- **Idea:** "fold" a fixed 2D grid into a 3D surface. Replicate the codeword $z$
  $m$ times, concatenate each with a 2D grid point $(u, v)$ → MLP → 3D point.
  Two consecutive folding MLPs.
- ~7 % of the params of a fully-connected decoder; achieves high linear-SVM
  accuracy.
- **Loss:** Chamfer Distance (CD) or Earth Mover's Distance (EMD).
- **Assumption:** the 3D object lies on a 2D manifold (true for surfaces, not
  volumes).

### AtlasNet (Groueix et al., CVPR 2018)

- Extends FoldingNet: learn **multiple small patches (atlases)**, each decoded
  separately then merged. Better fit to complex non-disk-topology surfaces.

### TopNet (Tchapmi et al., CVPR 2019)

- [PDF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Tchapmi_TopNet_Structural_Point_Cloud_Decoder_CVPR_2019_paper.pdf)
- **Idea:** rooted tree decoder. Root = encoder feature vector; leaves =
  individual points; internal nodes = subsets of the point cloud. Generates
  topology + points jointly.
- Trained with Chamfer distance for shape completion.

### PCN — Point Completion Network (Yuan et al., CVPR 2018)

- Two-stage decoder: (1) coarse point cloud from $z$ via FC layers, (2) refine
  via a local folding network. The canonical shape-completion baseline.

### SeedFormer (Zhou et al., ECCV 2022)

- [PDF](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136630409.pdf)
- **Idea:** generate **patch seeds** (coarse point positions + features) from
  $z$, then upsample via a SeedFormer Upsampling Transformer (SUT) to the full
  cloud. Better than FoldingNet/PCN at preserving existing structure and
  recovering missing details.
- Modern SOTA-adjacent point-cloud completion architecture.

### Multi-Head Decoders (2025)

- [arXiv:2505.19057](https://arxiv.org/html/2505.19057v1) — "Less is More":
  shows a _single_ decoder head is often sufficient, and deeper decoders don't
  always help; multi-head decoders give efficiency gains.

## Loss functions (the hard part)

- **Chamfer Distance (CD):**
  $\text{CD}(S, \hat{S}) = \max\left(\sum_{x \in S} \min_{\hat{x} \in \hat{S}} \|x - \hat{x}\|^2, \sum_{\hat{x} \in \hat{S}} \min_{x \in S} \|\hat{x} - x\|^2\right)$.
  Cheap but causes **local clustering** of reconstructed points.
- **Earth Mover's Distance (EMD):** bijection-based, more accurate but $O(N^2)$
  to compute exactly; approximations exist.
- **Density-Aware Chamfer Distance:** up-weights sparse regions — relevant to
  PointGFAE's density prior (see `paper/pointgfae/`).

## Relevance to the EB-JEPA hackathon

For an action-conditioned Point-JEPA with a generative decoder head:

1. **Flow-addition (PointWorld-style) is the default.** Only use a generative
   decoder below if you need to imagine clouds from pure latents
   (goal-conditioned planning, counterfactual rollouts).
2. **FoldingNet is the lightest generative decoder** — ~7 % of FC-decoder
   params, integrates cleanly as a head on top of the JEPA encoder/predictor
   latent. Good first try.
3. **SeedFormer is the strongest** if you need high-fidelity imagination — but
   heavier.
4. **Loss choice matters.** If you train a generative decoder alongside the JEPA
   latent loss, use density-aware CD to avoid the clustering failure mode,
   especially under the `data.rotate=so3` view-invariance study where
   sparse-region geometry is the bottleneck.
5. **Decoder-free vs decoder debate.** LeCun's JEPA position (see
   `paper/iwm-image-world-models/`, `paper/reconstruction-or-semantics/`) is
   that the decoder is _not needed_ for representation learning — latent
   prediction suffices. PointWorld's near-trivial decoder is the pragmatic
   middle ground: a decoder so simple it's almost free, used only to render
   rollouts for planning, not to train the encoder. This is the recommended
   stance for the hackathon.

## Caveats / open threads

- All these decoders are **input-space generative** — they reconstruct point
  coordinates, which LeCun's JEPA line argues wastes capacity on unpredictable
  detail. Use them only for the planning rollout renderer, not as the SSL
  objective.
- CD vs EMD is an old, unresolved trade-off; density-aware variants help but add
  hyperparameters.
- Most were designed for _single-object_ shape completion; scene-scale point
  clouds (PointWorld's regime) need flow-addition or seed-based upsampling, not
  single-grid folding.
