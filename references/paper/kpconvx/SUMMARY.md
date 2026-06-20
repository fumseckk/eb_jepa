# KPConvX — Modernizing Kernel Point Convolution with Kernel Attention

**Authors:** Hugues Thomas et al. (Apple) **Venue/Year:** CVPR 2024 **PDF:**
https://openaccess.thecvf.com/content/CVPR2024/papers/Thomas_KPConvX_Modernizing_Kernel_Point_Convolution_with_Kernel_Attention_CVPR_2024_paper.pdf
**Apple ML research post:** https://machinelearning.apple.com/research/kpconvx

## TL;DR

Revival of KPConv for the modern era. Two new operators:

- **KPConvD** — depthwise KPConv, lighter, enables deeper architectures.
- **KPConvX** — scales the depthwise KPConvD weights with **kernel attention**
  values (feature-driven modulation of the geometric kernel), without extra
  parameters.

Paired with a modern architecture + training strategy, KPConvX **outperforms
current SOTA** (Point Transformer V2/V3, PointNeXt) on ScanObjectNN, Scannetv2,
and S3DIS.

## Why it matters for this hackathon

- This is the **recommended** KPConv variant if you want to swap the PointNet
  encoder in `examples/pointcloud/` for a kernel-point encoder: modern training
  recipe, PyTorch-native, beats PTv2/PointNeXt, and the "kernel attention" is
  conceptually adjacent to JEPA's latent prediction.
- The deformable + attention design is a natural fit for a JEPA encoder whose
  job is to produce geometry-aware, predictable latent features.

## Caveats / open threads

- Code/models released by Apple; check license before bundling.
- Heavier than the minimal PointNet the track README asks for — use as an
  ablation / upgrade path.
