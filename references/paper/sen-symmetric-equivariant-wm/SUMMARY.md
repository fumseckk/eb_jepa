# SEN — Learning Symmetric Embeddings for Equivariant World Models

**Authors:** Minseob Park, Hyeongseok Jeon, Jihun Yun, Jihyeon Yeo, Seulki Park,
Jongmin Bae, Sung Ju Hwang (KAIST) **Venue/Year:** ICML 2022
([PMLR v162/park22a](https://proceedings.mlr.press/v162/park22a/park22a.pdf))

## TL;DR

A **pre-JEPA (2022) equivariant world model** that learns _symmetric_
(group-equivariant) latent embeddings and a $G$-equivariant transition model $T$
in latent space. The group action $\rho_Z(g)$ on the latent is _learned_, and
applying it to the embedding produces the correct transformed observation when
decoded. Demonstrated on 3D Teapot (SO(3) rotations) and Reacher — the latent
rotations align with the object's 3D pose, and decoding $\rho_Z(g) \cdot z$
renders the rotated object.

This is the closest existing precedent for "rotate the latent by a group action,
decode to a transformed observation" — but on images, not point clouds, and not
JEPA (uses a reconstruction + equivariance loss, not latent prediction against
an EMA target).

## Problem & motivation

- Standard world models / SSL learn _invariant_ representations, discarding pose
  information needed for control and equivariant downstream tasks.
- Want a representation that is $G$-equivariant:
  $f(g \cdot x) = \rho_Z(g) \cdot f(x)$, where the group action $\rho_Z$ on the
  latent is (learnably) consistent with the physical group action.

## Method

### Equivariant transition in latent space

Factorize the group representation on state × action as latent state transform
$\rho_Z(g) \cdot E(s)$ and action transform $\rho_A(g; s) \cdot a$. The
transition model is $G$-equivariant:
$$\rho_Z(g) \cdot T(E(s), a) = T(\rho_Z(g) \cdot E(s),\; \rho_A(g; s) \cdot a), \quad \forall g \in G$$

### Learned latent group action

- $\rho_Z(g)$ is _learned_ (not hard-coded), so the latent picks its own
  canonical frame.
- Applying $\rho_Z(g)$ to an embedding $z$ and decoding yields the transformed
  observation.

### Decoding the transformed latent

- A decoder maps $z \to \hat{o}$ (pixel image).
- For 3D Blocks and Reacher, a separate decoder is trained after freezing the
  model to decode $z$ into pixel space; Figures 9–10 show the learned $\rho_S$
  (input-space group action) corresponds to the latent $\rho_Z$.

## Key results

- 3D Teapot: traversing rotations in latent space ($\rho_Z(g) \cdot z$) and
  decoding produces correctly rotated teapots — the latent encodes 3D pose and
  the group action is consistent.
- Reacher: latent embeddings cluster by pose; equivariant model generalizes
  better than invariant baselines on hard-hit tasks.
- Surprisingly robust even when the input-space group action $\rho_S$ is
  inaccurate (skewed perspective) — equivariance doesn't hurt in-distribution,
  only constrains OOD extrapolation.

## Relevance to the EB-JEPA hackathon — "rotate the latent, decode the rotated point cloud"

SEN is the conceptual template for the user's requested pattern, minus the JEPA
objective:

```
point cloud X ──encoder E──► z
                              │
      action g (rotation)     │
                              ▼
                      ρ_Z(g) · z   (learned group action on the latent)
                              │
                              ▼
                   decoder D ──► transformed point cloud X̂'
```

**Direct mapping:**

- $E$ → EB-JEPA encoder (make it SE(3)-equivariant: SE(3)-Transformer / EPN /
  E2PN).
- $\rho_Z(g) \cdot z$ → the action-conditioning mechanism — "rotate/scale the
  latent."
- $T(E(s), a)$ → EB-JEPA predictor $P_\theta(z, a)$, made $G$-equivariant.
- $D$ → point-cloud decoder (FoldingNet/SeedFormer, or PointWorld
  flow-addition).

**The key insight for the hackathon:** SEN shows the group action on the latent
can be _learned_ ($\rho_Z$ doesn't need to be the canonical representation), and
decoding the transformed latent yields the correctly transformed observation.
For point clouds, if the encoder is SE(3)-equivariant, then $\rho_Z(g)$ is
_analytically known_ (it's the same rotation applied to the equivariant
features), so you don't even need to learn $\rho_Z$ — you can apply the action
directly in latent space and decode.

**To make it a JEPA:** replace SEN's reconstruction + equivariance loss with (i)
latent L1 prediction against an EMA target encoder, (ii) VICReg/SIGReg
anti-collapse, (iii) a point-cloud decoder used only for the ID head. This
combination is the unfilled niche.

## Caveats / open threads

- Not JEPA: uses reconstruction + equivariance loss, not latent prediction
  against an EMA target. Combining with VICReg/SIGReg is untested.
- Image domain (3D Teapot renders, Reacher pixels), not raw point clouds.
- $\rho_Z$ is learned, which adds parameters; for point clouds with an
  SE(3)-equivariant encoder, $\rho_Z$ is analytically determined, simplifying
  the design.
- Decoder is trained separately (frozen encoder), not jointly — a JEPA would
  jointly train but use the decoder only as a grounding head.
