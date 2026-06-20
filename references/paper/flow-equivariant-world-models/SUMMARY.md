# Flow Equivariant World Models (FloWM) — Memory for Partially Observed Dynamic Environments

**Authors:** Hansen Lillemark, Benhao Huang, Fangneng Zhan, Yilun Du, T.
Anderson Keller (Harvard, UCSD, CMU) **Venue/Year:** arXiv 3 Jan 2026,
[2601.01075](https://arxiv.org/abs/2601.01075); project page
[flowequivariantworldmodels.github.io](https://flowequivariantworldmodels.github.io)

## TL;DR

A **generative** (not JEPA) world model whose entire encode→update→flow→decode
pipeline is **equivariant to Lie-group flows** — both agent self-motion (rigid
transforms) and external object dynamics (continuous flows). Maintains a
structured latent memory indexed by _velocity channels_ $v \in \mathfrak{g}$
(the Lie algebra); at each step the memory is rolled by composing the agent
action transform $T_{a_t}^{-1}$ with the internal flow $\psi_1(v)$. The decoder
renders the latent back to an observation. Beats diffusion baselines on
long-horizon rollout accuracy and data efficiency in partially observed 2D
MNIST-World and 3D Blockworld.

This is the closest existing instantiation of "action = a geometric transform of
the latent, decode back to observation" — except it operates on image/voxel
observations, not raw point clouds, and is generative rather than JEPA.

## Problem & motivation

- Standard world models (Dreamer, diffusion video WMs) don't enforce symmetry:
  small rotations of the input produce unpredictable latent changes, hurting
  long-horizon rollout accuracy and generalization under partial observability.
- Want: a world model whose latent dynamics _commute_ with the underlying
  spatiotemporal symmetry group, so rollouts stay accurate over long horizons
  and the model generalizes to unseen poses.

## Method

### Equivariance goal

$$f(g \cdot x) = g \cdot f(x), \quad \forall g \in G$$ enforced at _every_
architectural stage (encode, update, flow, decode).

### Velocity-channel latent memory

- Latent state is a tensor indexed by discrete velocity channels
  $v \in V \subset \mathfrak{g}$ (the Lie algebra).
- Each channel evolves under a flow-induced "shift" in latent space (integer
  rolls for translation, permutations for 3D rigid motion).

### FloWM recurrence (the action-conditioned latent predictor)

$$\underbrace{h_{t+1}(v)}_{\text{next latent}} = \underbrace{T_{a_t}^{-1}}_{\text{self-action transform}} \cdot \underbrace{\psi_1(v)}_{\text{internal flow transform}} \cdot \underbrace{U_\theta[h_t(v),\, E_\theta[f_t, h_t](v)]}_{\text{update memory}}$$

- $E_\theta$ — equivariant encoder (ViT in 3D).
- $U_\theta$ — equivariant memory update (gated concat + equivariant linear
  map).
- $T_{a_t}^{-1}$ — the **agent action as a group transform on the latent**
  (rotation / translation of the memory map). This is exactly "the action
  rotates/translates the latent."
- $\psi_1(v)$ — internal/external flow (object motion), composed with the agent
  action.
- Decoder renders $h_{t+1} \to \hat{o}_{t+1}$.

### Closure property

When all maps are equivariant and the initial state is consistent across
velocity channels, the latent memory remains flow-equivariant for arbitrarily
long rollouts — formally preventing drift under sequences of self-motions.

## Key results

- 2D MNIST-World (partially observed): FloWM beats non-equivariant RNN and
  diffusion baselines on rollout PSNR, with faster training convergence.
- 3D Blockworld (dynamic, textured, static splits): ViT encoder + 3D FloWM
  recurrence; stable long-horizon rollouts where baselines diverge.
- Ablations: removing the action transform $T_{a_t}^{-1}$ or the internal flow
  $\psi_1$ degrades rollout error substantially — both composition terms are
  load-bearing.

## Relevance to the EB-JEPA hackathon — the equivariant-JEPA blueprint

FloWM is the paper that most directly validates the user's "action =
rotate/scale the latent, decode to observation" pattern, with one caveat: it's
**generative** (decodes to pixels/voxels as the training objective), not JEPA
(predict-in-latent-space).

**Direct port to an equivariant AC-Point-JEPA:**

```
point cloud X_t ──equivariant encoder──► z_t  (SE(3)-equivariant latent)
                                             │
          action a_t = (R, t, s)  ───────────┤  (rotate / translate / scale)
                                             ▼
                                   T_{a_t}^{-1} · z_t   (group action on latent)
                                             │
                                             ▼  (+ optional internal flow)
                                          z_{t+1}   (predicted latent)
                                             │
                  ┌──────────────────────────┘
                  ▼                          ▼
        (JEPA path: L1 to EMA target)   (decoder path: decode → point cloud)
```

- Replace FloWM's generative decoder-with-L2 objective with the JEPA objective:
  predict $\hat{z}_{t+1} = P_\theta(z_t, a_t)$ and match the EMA target
  encoder's $z_{t+1}^-$.
- Keep the **equivariant encoder** (SE(3)-Transformer / Equivariant Point
  Network / E2PN — see `paper/se3-equivariant-pointcloud-backbones/`) so that
  $E_\theta(g \cdot X) = g \cdot E_\theta(X)$.
- Keep the **group action on the latent** $T_{a_t}^{-1}$ as the
  action-conditioning mechanism — this is the "rotate/scale the latent" the user
  described.
- Add a **point-cloud decoder** $D_\phi(\hat{z}_{t+1}) \to \hat{X}_{t+1}$
  (FoldingNet/SeedFormer, or PointWorld flow-addition) used _only_ for the
  inverse-dynamics head / visualization, not as the training objective.

**Why this matters:** FloWM proves the equivariant encode→transform→decode
recurrence is stable over long horizons. Combining its equivariance machinery
with EB-JEPA's latent-prediction objective

- VICReg/SIGReg anti-collapse would yield exactly the "action-conditioned
  equivariant JEPA on point clouds with a point-cloud decoder" the user is
  asking for.

## Caveats / open threads

- Generative, not JEPA: the decoder loss is part of training, so removing it (to
  go pure JEPA) is untested. The closure property assumes all stages are
  equivariant; a non-equivariant decoder (FoldingNet is not SE(3)-equivariant)
  would break the guarantee — an open design question.
- Operates on image/voxel observations, not raw point clouds; the 3D Blockworld
  uses a ViT over rendered views, not a point-cloud encoder.
- Velocity-channel memory is heavier than a flat global latent; trade-off vs a
  KPConv global latent is an open ablation.
- External flow $\psi_1(v)$ is learned, not given; for the hackathon's
  rigid-action case (rotate/ scale), the action transform $T_{a_t}^{-1}$ is
  known analytically and $\psi_1$ can be dropped.
