# 4D Latent World Model for Robot Planning

**Venue/Year:** Under review, ICLR 2026 **PDF:**
[openreview.net/pdf?id=d19b200c60835c287e29c6d20ec394f93038c007](https://openreview.net/pdf/d19b200c60835c287e29c6d20ec394f93038c007.pdf)

## TL;DR

**The closest match to the user's exact request**: an action-conditioned world
model that (a) predicts in a **structured 3D latent space** (sparse voxel grids,
not pixels, not raw points), and (b) **decodes the predicted latents into point
clouds** (or rendered views) which feed a goal-conditioned inverse-dynamics
module for robot control. This is the "JEPA-style latent prediction +
decoder-to-point-cloud" pattern, instantiated for 4D embodied planning.

## Problem & motivation

- Prevailing world models predict sequences of 2D frames per viewpoint; fusing
  per-view models doesn't give true 3D consistency.
- Pure video diffusion is pixel-space → wastes capacity on appearance detail
  that doesn't matter for control, and produces visually plausible but
  action-inconsistent futures.
- Want: (i) 3D-consistent state, (ii) action-conditioned dynamics in latent
  space, (iii) decodable into explicit 3D formats (point clouds / 3D Gaussians)
  for downstream inverse dynamics.

## Method

### 1. Structured 3D latent state

- Encode the scene from multi-view images into a sequence of **sparse voxel
  grids**, where each active voxel holds a compact feature vector.
- This gives explicit 3D spatial bias + the computational efficiency and
  semantic abstraction of a low-dim latent (à la Latent Diffusion's
  spatially-aware 2D feature maps, but in 3D).

### 2. 4D Latent World Model (the dynamics predictor)

Predicts future 3D latents conditioned on current latent + text instruction,
using two components:

- **Single Dynamics Model** — coarse structural changes.
- **Latent Generator** — detailed features (a latent generative head). This
  split mirrors the "predict the big structure cheaply, generate fine detail
  separately" pattern.

### 3. Decode to explicit 3D formats

The predicted latents are **decoded into point clouds OR rendered views (3D
Gaussian Splatting)**. This is the explicit "decoder that decodes to a point
cloud" component.

### 4. Goal-conditioned inverse dynamics

- Decode $z_t, z_{t+1}$ into lighter point clouds $pc_t, pc_{t+1}$.
- ID module $ID(s_1, \dots, s_H | z_t, z_{t+1})$ outputs absolute joint
  positions (low-level robot commands) that move the robot from $z_t$ to the
  subgoal $z_{t+1}$.
- The decoder is used only here, to ground the latent subgoal into geometry for
  the action head — exactly the "decoder as renderer for the planner, not as the
  training objective" pattern.

## Key results

- SOTA in 3D-aware generative modeling benchmarks.
- Significant improvements in downstream robotic planning over 2D video world
  models.
- Decoded point clouds give the ID module geometrically grounded subgoals,
  especially effective for fine-grained 3D-aware tasks.

## Relevance to the EB-JEPA hackathon — the exact recipe

This paper is the most direct instantiation of the user's requested
architecture:

```
multi-view RGB-D → 3D latent z_t
                     │
   text/action a_t   ▼
              ┌──────────────┐
              │ 4D Latent WM │  ← predicts in latent space (JEPA-style)
              │  (dynamics)  │
              └──────────────┘
                     │
                     ▼
                ẑ_{t+1} (predicted latent)
                     │
                     ▼  ← DECODER
              ┌──────────────┐
              │  point cloud │  ← decodes to a point cloud
              │   pc_{t+1}   │
              └──────────────┘
                     │
                     ▼
              inverse dynamics → robot actions
```

**Porting to eb_jepa:**

1. Replace the sparse-voxel latent with the EB-JEPA encoder latent
   $z_t = E_\theta(s_t)$ over a point-cloud state $s_t$ (KPConv/PTv3 encoder).
2. Replace the Single Dynamics Model + Latent Generator with the EB-JEPA
   predictor $P_\theta(z_t, a_t) \to \hat{z}_{t+1}$ + VICReg/SIGReg
   anti-collapse.
3. Keep the **point-cloud decoder** $D_\phi(\hat{z}_{t+1}) \to pc_{t+1}$ — this
   is the "decoder that decodes to a point cloud" the user asked for. Use
   FoldingNet/SeedFormer (see `paper/point-cloud-decoders/`) or the PointWorld
   flow-addition trick (`paper/pointworld/`).
4. Keep the **goal-conditioned inverse dynamics** head — this is exactly the IDM
   loss `examples/ac_video_jepa/` uses to prevent collapse.

The split between "predict in latent space (training objective)" and "decode to
point cloud (only for the inverse-dynamics head / planning rollout renderer)" is
precisely the EB-JEPA philosophy: the decoder is a near-trivial grounding head,
not the SSL objective.

## Caveats / open threads

- Under review (ICLR 2026) — final version may differ.
- Sparse-voxel latent is heavier than a flat global vector; the trade-off vs a
  KPConv global latent is an open ablation.
- Uses text-conditioned dynamics (a text instruction), not raw continuous
  actions — for the hackathon you'd swap text for the action embedding
  (point-flow or SE(3) params).
- The Latent Generator (detailed-features head) leans generative; a pure JEPA
  variant would drop it and rely only on the latent L1 loss + decoder for
  grounding.
