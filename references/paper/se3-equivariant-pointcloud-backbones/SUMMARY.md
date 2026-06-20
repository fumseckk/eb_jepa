# SE(3)-Equivariant Point-Cloud Backbones — for an Equivariant AC-Point-JEPA

**Papers:** SE(3)-Transformer (Fuchs et al., NeurIPS 2020) · Equivariant Point
Network / EPN (Chen et al., CVPR 2021) · E2PN (Lee et al., 2024) · Equivariant
Transformer for Point Cloud Registration (2024)

## TL;DR

A family of point-cloud encoders that are **exactly SE(3)-equivariant by
construction**: rotating the input point cloud rotates the output features by
the same group action, $E(g \cdot X) = g \cdot E(X)$. This is the _encoder_
ingredient that makes "rotate the latent as the action" analytic rather than
learned. Surveyed here as candidate backbones to replace the minimal PointNet in
`examples/pointcloud/` when building an equivariant action-conditioned JEPA.

## Why this matters for the AC-Point-JEPA design

If the encoder $E_\theta$ is SE(3)-equivariant, then for a rigid action
$a = (R, t)$: $$E_\theta(R \cdot X + t) = R \cdot E_\theta(X) + t'$$ The group
action on the latent $\rho_Z(g)$ is **the same rotation applied to the
features** — no need to learn it (cf. SEN,
`paper/sen-symmetric-equivariant-wm/`, which learns $\rho_Z$ because its encoder
is not equivariant). This means:

- The "action = rotate the latent" step is a **free, exact operation** (just
  apply $R$ to the equivariant features).
- The predictor $P_\theta(z, a)$ can be a simple equivariant map, or even the
  analytic group action itself for rigid transforms.
- The decoder $D_\phi(z) \to \hat{X}$ decodes the rotated latent back to a
  rotated point cloud.

## The backbones

### SE(3)-Transformer (Fuchs et al., NeurIPS 2020)

- Self-attention for 3D point clouds/graphs, **exactly SE(3)-equivariant**.
- Uses irreducible representations of SO(3) and tensor-field attention;
  guarantees $f(g \cdot x) = g \cdot f(x)$ for continuous 3D roto-translations.
- Competitive on ScanObjectNN, QM9; robust under input rotations (N-body
  simulation).
- **Pros:** exact equivariance, attention-based (good for long-range). **Cons:**
  heavier than KPConv; higher-dimensional feature spaces (irreps).

### Equivariant Point Network / EPN (Chen et al., CVPR 2021)

- SE(3)-separable convolution + attention over equivariant features.
- Defines continuous features
  $F(x_i, g_j) : \mathbb{R}^3 \times SO(3) \to \mathbb{R}^D$; the convolution is
  proved equivariant to SE(3).
- **Pros:** more expressive than invariant features (retains pose info
  throughout layers); permutation-invariant. **Cons:** computing over the full
  SE(3) space is costly; uses separable convolution to mitigate.

### E2PN / Equivariant Transformer for Registration (Lee et al. 2024)

- Extends EPN with equivariant self-attention (ESA) and cross-attention modules
  for point-cloud registration.
- Extracts both equivariant and invariant features at multiple resolutions.
- **Pros:** designed for the registration task (closest to "rotate the cloud to
  align with a target"); cross-attention between two clouds. **Cons:**
  registration-specific; may need adaptation for world-model rollout.

## Relevance to the EB-JEPA hackathon — the equivariant encoder slot

For the `examples/pointcloud/` track, the current encoder is a minimal PointNet
(per the README). To build the user's "action-conditioned JEPA where the action
rotates/scales the point cloud and a decoder decodes back to a point cloud," the
recommended stack:

```
point cloud X_t
     │
     ▼  SE(3)-equivariant encoder (SE(3)-Transformer / EPN / E2PN)
  z_t  (equivariant features — rotating X_t rotates z_t by the same R)
     │
     │  action a_t = (R, t, s)  ── analytic group action on z_t
     ▼
  z_t' = R · z_t + t'            (the "rotated/scaled latent" — free, exact)
     │
     ▼  EB-JEPA predictor P_θ (optional learned residual for non-rigid dynamics)
  ẑ_{t+1}
     │
     ├─► JEPA loss: L1(ẑ_{t+1}, EMA_target(X_{t+1}))  + VICReg/SIGReg
     │
     └─► decoder D_φ(ẑ_{t+1}) → point cloud X̂_{t+1}   (FoldingNet/SeedFormer,
                                                         or PointWorld flow-addition)
                                                         used for ID head / viz only
```

**Why this is the missing piece:** the existing AC-JEPA lineage (IWM, seq-JEPA,
V-JEPA 2-AC) is image/video; the point-cloud JEPAs (Point-JEPA, 3D-JEPA) are
action-free; PointWorld is generative in point-flow space; FloWM and SEN are
equivariant world models but generative and image-domain. An **SE(3)-equivariant
encoder** + **EB-JEPA latent objective** + **point-cloud decoder-as-renderer**
fills the gap: the action (rotate/scale) is applied analytically to the
equivariant latent, the predictor learns any residual dynamics, and the decoder
grounds the latent back to a point cloud for the planner.

## Caveats / open threads

- Equivariant encoders are heavier and higher-dimensional (irreps) than
  PointNet/KPConv; the trade-off vs a non-equivariant encoder + learned $\rho_Z$
  (SEN-style) is an open ablation.
- FoldingNet/SeedFormer decoders are **not** SE(3)-equivariant — using them as
  the decoder breaks the end-to-end equivariance guarantee (cf. FloWM's closure
  property). Options: (a) accept the break (decoder is only for the ID head, not
  the training objective), (b) use an equivariant decoder (e.g., equivariant
  FoldingNet), or (c) use PointWorld's flow-addition decoder which is
  equivariant by construction (flow is a vector field).
- For non-rigid actions (articulated, deformable), the analytic group action is
  insufficient; the predictor $P_\theta$ must learn the residual, moving toward
  FloWM's learned $\psi_1(v)$.
- ScanObjectNN/QM9 are the typical benchmarks; ModelNet40 (the
  `examples/pointcloud/` dataset) is cleaner but less challenging — equivariance
  gains may be smaller.
