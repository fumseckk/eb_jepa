# Planning with Learned Dynamic Model for Unsupervised Point Cloud Registration

**Venue/Year:** IJCAI 2021 **PDF:**
[ijcai.org/proceedings/2021/0107](https://www.ijcai.org/proceedings/2021/0107.pdf)

## TL;DR

A small, clean, **pre-JEPA** precedent that already has every component the user
asked for:

- **Action-conditioned latent dynamics** on point clouds.
- An **encoder-decoder** that lifts a point cloud to a latent and decodes back
  to a point cloud (via Chamfer reconstruction).
- A **transformation network** that predicts the latent of the _transformed_
  cloud conditioned on the rigid action — the action "rescales/rotates" the
  cloud, exactly the user's phrasing.
- **CEM planning** over the action space to reach a goal latent — the same
  planner EB-JEPA uses.

It is essentially a 2021 proto-JEPA for point clouds: predict the latent of the
action-transformed cloud, decode to a point cloud, plan with CEM.

## Problem & motivation

Point cloud registration framed as an MDP: state = source point cloud, action =
rigid transformation, goal = align source to target. Learn a latent dynamic
model unsupervised, then plan with CEM to maximize alignment reward (predicted
by an evaluation network).

## Method

### 1. Auto-encoder (encoder + decoder → point cloud)

- Encoder $\phi_{enc}$ maps point cloud $X \to z$ (latent state).
- Decoder $\phi_{dec}$ maps $z \to \hat{X}$ (point cloud).
- Trained with **Chamfer distance** reconstruction loss:
  $$\mathcal{L}_{rec}(\phi, \omega) = \frac{1}{|S|}\sum_{\langle X, X', Y\rangle \in S} L(X) + L(X') + L(Y)$$
  $$L(X) = \text{Dcd}(\phi_{dec}^\omega(\phi_{enc}^\phi(X)), X), \quad \text{Dcd} = \text{Chamfer}$$

### 2. Transformation network $f_\theta^z$ (the action-conditioned latent predictor)

- Takes the source latent $z_X$ **and the action** (rigid transformation
  params).
- Predicts the latent of the transformed source cloud
  $\hat{z}_{X'} = f_\theta^z(z_X, a)$.
- This is the **action-conditioned latent predictor** — the JEPA predictor role.
  The action is literally a rescaling/rotation of the point cloud.

### 3. Evaluation network (the reward / cost head)

- Predicts alignment precision between the transformed source and the target, in
  feature space.
- Used as the planning reward.

### 4. CEM planning

- Iteratively sample action trajectories, roll out the latent dynamics, score
  with the evaluation network, keep elite samples, update the sampling
  distribution.
- Same algorithm EB-JEPA / ac_video_jepa uses (CEM/MPPI), but in 2021 and on
  point clouds.

## Relevance to the EB-JEPA hackathon — the minimal viable pattern

This is the **smallest, most portable** recipe for "action-conditioned JEPA +
point-cloud decoder":

```
source cloud X ──encoder──► z_X
                              │
              action a (rigid)│
                              ▼
                      f_θ^z(z_X, a) ──► ẑ_X'   (action-conditioned latent prediction)
                              │                 │
                              ▼                 ▼
                       (decode, optional)   compare to z_target (EMA / target encoder)
                              │
                              ▼
                          point cloud X̂'  (decoder → point cloud, for visualization / ID head)
                              │
                              ▼
                       CEM plan over a to reach goal latent
```

**Direct mapping to EB-JEPA:**

- $\phi_{enc}$ → EB-JEPA encoder $E_\theta$ (KPConv / PointNet).
- $f_\theta^z$ → EB-JEPA predictor $P_\theta(z, a)$.
- $\phi_{dec}$ → the point-cloud decoder head (FoldingNet/SeedFormer, or
  PointWorld flow-addition).
- Evaluation network → the IDM / planning cost head (cf.
  `examples/ac_video_jepa`).
- CEM → `eb_jepa/planning.py` CEM (already implemented).

**Why this matters:** it shows the pattern works at small scale (single-object,
rigid actions, IJCAI 2021 compute). For a hackathon starting from
`examples/pointcloud/`, this is the closest existing precedent to extend into an
action-conditioned Point-JEPA without needing PointWorld-scale data or the 4D
Latent World Model's sparse-voxel machinery.

## Caveats / open threads

- Pre-JEPA (2021): no EMA target encoder, no VICReg/SIGReg — uses Chamfer
  reconstruction as the anti-collapse mechanism (the GLP-style argument that
  reconstruction prevents collapse). Adding VICReg/SIGReg (eb_jepa core) would
  let you drop the reconstruction loss and make it a true JEPA.
- Single-object registration only; not scene-scale.
- Rigid actions only (rotation + translation + scale); extending to articulated
  / deformable actions is open.
- The "evaluation network" is task-specific (alignment precision); a JEPA would
  replace it with a goal-latent L1 cost (cf. V-JEPA 2-AC, LeWM).
