# PointWorld — Scaling 3D World Models for In-The-Wild Robotic Manipulation

**Authors:** Wenlong Huang (Stanford), Yu-Wei Chao, Arsalan Mousavian, Ming-Yu
Liu, Dieter Fox, Kaichun Mo (NVIDIA), Li Fei-Fei (Stanford) **Venue/Year:** CVPR
2026 (Highlight) — Best Paper at the E2E3D Workshop at CVPR 2026 **Project:**
[point-world.github.io](https://point-world.github.io) · **arXiv:**
[2601.03782](https://arxiv.org/abs/2601.03782) · **Code:**
[NVlabs/PointWorld](https://github.com/NVlabs/PointWorld)

## TL;DR

The most directly relevant paper to "action-conditioned point cloud with a
decoder that decodes to a point cloud." PointWorld is a **large pre-trained 3D
world model** that, given RGB-D capture(s) and a sequence of robot actions,
**predicts full-scene 3D point flow** — i.e. per-point 3D displacements of the
_next_ point cloud. The key trick: **represent both state and action as 3D point
flows in a shared 3D space**, so the action is embodiment-agnostic. ~2M
trajectories, 500 h, single-arm Franka + bimanual humanoid; real-time (0.1 s)
inference; deployed on a real Franka via MPPI-MPC for pushing, deformable
manipulation, and bimanual tasks.

## Problem & motivation

- Humans glance at a scene + imagine an action → anticipate how the 3D world
  responds. Robots need the same for manipulation.
- Prior world models predict in pixel/image space (Cosmos, PEVA, Dreamer V3) or
  in embodiment- specific action spaces (joint positions), which (a) wastes
  capacity on unpredictable texture, (b) doesn't transfer across embodiments.
- **Insight:** unify state and action in 3D — represent the action itself as a
  3D point flow over the robot's gripper/links, then predict the resulting scene
  point flow. The "decoder" is just "add the predicted flow to the current point
  cloud" — the next-frame point cloud falls out for free.

## Method

### 3.1 Problem formulation — action-conditioned 3D point flow

- **State** $s_t$: a full-scene point cloud $\{x_i \in \mathbb{R}^3\}$ from
  calibrated RGB-D.
- **Action** $a_{t:t+H-1}$: a temporal sequence of **robot point-flow actions**
  — for each future step, the 3D displacement of each point on the robot's
  gripper/links (300–500 points per gripper), computed from the URDF +
  joint-space action commands.
- **Target** $\hat{s}_{t+1:t+H}$: the future scene point clouds, equivalently
  the per-point scene flow $d_i = x^{(t+1)}_i - x^{(t)}_i$.
- The model predicts $d_i$; the next-frame point cloud is decoded trivially as
  $x^{(t+1)}_i = x^{(t)}_i + d_i$.

### 3.2 Architecture

- **Concatenate** the static scene points $\mathbf{s}_t$ with the time-stacked
  robot point-flow actions $\mathbf{a}_{t:t+H-1}$ into a single point cloud.
- **Scene points** are featurized with a **frozen DINOv3** encoder (project to
  2D views → features).
- **Robot points** are featurized with **temporal embeddings**.
- A **Point Transformer v3 (PTv3)** backbone processes the concatenated cloud
  and predicts the full-scene 3D point flow. (They deliberately use SOTA point
  backbones rather than a custom arch, to distill general scaling principles.)
- The "decoder to a point cloud" is the **flow-addition step**: predicted flow +
  current points = next-frame point cloud. This is the cleanest possible
  decoder.

### 3.3 Action inference via MPC

- Wrap PointWorld in **MPPI** (sampling-based planner): sample $K$ action
  perturbations $\ell_{1:K}$ with cubic-spline time-correlated noise, add to a
  nominal end-effector trajectory, build robot point-flow actions
  $a^{(\ell)}_{1:T}$, roll out scene flows with PointWorld, accumulate a
  trajectory cost $J^{(\ell)}$, refine the mean with elite-sample weighting.
- Plans a sequence of $T$ end-effector pose targets in **SE(3)**.
- Real-time: 0.1 s inference → closed-loop MPC on a real Franka.

### Action representation ablations (key finding)

Point-flow actions beat all low-dim alternatives:

1. whole-body point flows (same #points, sparser coverage),
2. whole-body point clouds (2000 points, similar density),
3. 6-DoF end-effector pose + gripper openness,
4. joint positions + gripper openness. → Gripper point flows balance contact
   reasoning + cross-embodiment transfer.

## Key results

- ~2M trajectories / 500 h across single-arm Franka (DROID) + bimanual humanoid
  (B1K).
- Real-world Franka: rigid-body pushing, deformable manipulation, bimanual tasks
  — **single pre-trained checkpoint, zero-shot**.
- Real-time 0.1 s inference → closed-loop MPC.
- Best Paper at E2E3D Workshop @ CVPR 2026.

## Relevance to the EB-JEPA hackathon — the action-conditioned point-cloud recipe

This is the **closest analogue** to `examples/ac_video_jepa/` but for point
clouds, and it answers both halves of the request: "action conditioning" +
"decoder that decodes to a point cloud."

1. **Action = 3D point flow (rescaling/transforming the cloud).** The user's
   "rescaling point cloud based on an action" maps directly onto PointWorld's
   formulation. An action is _literally_ a per-point 3D displacement field
   applied to the cloud; for a global rescale/rotate action, the flow is
   $d_i = (R \cdot s \cdot x_i) - x_i$ where $R$ is rotation, $s$ is scale. The
   JEPA predictor then learns the latent→flow mapping.
2. **Decoder = flow addition.** The "decoder that decodes to a point cloud" is
   the trivial $x^{(t+1)} = x^{(t)} + \text{flow}(x^{(t)}, a)$. This avoids the
   entire FoldingNet/AtlasNet complexity (see `paper/point-cloud-decoders/`) and
   is exactly how PointWorld gets a real-time next-frame cloud.
3. **Recipe for an action-conditioned Point-JEPA.** Combine PointWorld's
   state/action representation with the EB-JEPA predictor + VICReg/SIGReg
   anti-collapse:
   - Encoder $E_\theta$: PointNet/KPConv/PTv3 → latent $z_t = E_\theta(s_t)$.
   - Predictor $P_\theta$: takes $(z_t, a_{t:t+H-1})$, predicts
     $\hat{z}_{t+1:t+H}$.
   - Anti-collapse: VICReg/SIGReg on $z$ (eb_jepa core) + IDM loss
     (ac_video_jepa recipe) to avoid the spurious-correlation collapse Sobal et
     al. identified.
   - **Optional decoder head**: a point-flow decoder
     $D_\phi(\hat{z}_{t+1}) \to \hat{d}_{t+1}$ trained with Chamfer/flow-L2
     against the ground-truth next cloud — this is the "decode to a point cloud"
     half. PointWorld shows this decoder can be near-trivial (flow addition) if
     the latent is geometry-aware; if you want a generative decoder, use
     FoldingNet/SeedFormer (see `paper/point-cloud-decoders/`).
4. **Planning.** PointWorld's MPPI-over-SE(3)-actions is the direct 3D analogue
   of `examples/ac_video_jepa`'s MPPI/CEM-over-2D-actions. The
   `eb_jepa/planning.py` MPPI/CEM code can be reused almost verbatim — just swap
   the action space from 2D to SE(3) point flow.

## Caveats / open threads

- **Not a JEPA** — PointWorld predicts in _point/flow space_ (generative), not
  in latent space. It is the generative counterpoint to a hypothetical
  "AC-Point-JEPA." The hackathon-relevant move is to lift PointWorld's
  state/action representation into EB-JEPA's latent-prediction framework.
- Needs RGB-D + URDF for the action-to-flow conversion; pure point-cloud
  datasets (ModelNet40, ShapeNet) don't have actions — you'd need a dynamics
  dataset (DROID, B1K, or a simulated Two-Rooms-style point-cloud env).
- Frozen DINOv3 for scene features — heavy dependency; a KPConv/PTv3-only
  variant is an open ablation.
- Scale: 2M trajectories is far beyond the hackathon's compute budget — the
  recipe is what's portable, not the scale.
