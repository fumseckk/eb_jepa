# FlowDreamer — A RGB-D World Model with Flow-Based Motion Representations for Robot Manipulation

**Authors:** Jun Guo, Xiaojian Ma, Yikai Wang, Min Yang, Huaping Liu, Qing Li
**Venue/Year:** IEEE Robotics and Automation Letters (RA-L), 2026 **arXiv:**
[2505.10075](https://arxiv.org/abs/2505.10075)

## TL;DR

An action-conditioned RGB-D world model that uses **3D scene flow as the
explicit motion representation**. Two-stage: (1) a U-Net predicts 3D scene flow
from the current RGB-D + robot action, (2) a diffusion model predicts the future
RGB frame using the flow. The point-cloud variant uses a **FiLM-conditioned
MinkowskiNet** (sparse 3D conv) as the dynamics predictor. Beats baseline RGB-D
world models by +7% semantic similarity, +11% pixel quality, +6% success rate.

## Method

- **Stage 1 (dynamics):** conditional 2D U-Net with 4 down/up layers; action
  integrated via cross-attention (Stable-Diffusion-style). Predicts 3D scene
  flow $f_{t\to t+1}$.
- **Point-cloud baseline variant:** conditional MinkowskiNet (modified
  MinkUNet34B) with FiLM layers replacing batch-norm to inject the action
  condition — this is the action-conditioned point-cloud dynamics model.
- **Stage 2 (rendering):** diffusion model predicts future RGB using the flow.
- Scene flow in point clouds = displacement of point coordinates; RGB-D pixels →
  3D points via camera intrinsics.

## Key results

- +7% semantic similarity, +11% pixel quality, +6% success rate vs baseline
  RGB-D world models on RT-1, Language Table, VP2/RoboDesk/Robosuite.

## Relevance to the EB-JEPA hackathon

- The **FiLM-conditioned MinkowskiNet** is a drop-in action-conditioned
  point-cloud dynamics backbone — useful if you want a sparse-3D-conv
  alternative to KPConv/PTv3 for the predictor.
- Validates 3D scene flow as the right intermediate motion representation for
  action-conditioned point-cloud prediction (same conclusion as PointWorld, but
  smaller-scale and earlier).
- Generative (predicts flow + renders pixels), not JEPA — but the
  flow-prediction stage is the same latent-prediction-in-disguise pattern:
  predict the displacement field, then apply it.

## Caveats / open threads

- RGB-D / image-centric; the point-cloud variant is a baseline, not the main
  contribution.
- Diffusion renderer is heavy; for a hackathon, the PointWorld flow-addition
  decoder is lighter.
- FiLM conditioning is a simple action-injection mechanism, but cross-attention
  (as in the 2D path) often scales better for high-dim actions like point flows.
