# seq-JEPA — Autoregressive Predictive Learning of Invariant-Equivariant World Models

**Authors:** Hafez Ghaemi, Eilif B. Muller, Shahab Bakhtiari (Mila / Université
de Montréal) **Venue/Year:** NeurIPS 2025 **PDF:**
[proceedings.neurips.cc/.../2f63d2963526bdd9ff1b8bcc2dc9905a](https://proceedings.neurips.cc/paper_files/paper/2025/file/2f63d2963526bdd9ff1b8bcc2dc9905a-Paper-Conference.pdf)
· **OpenReview:** [MO1OLAKcsJ](https://openreview.net/pdf?id=MO1OLAKcsJ)

## TL;DR

An action-conditioned JEPA that **automatically disentangles invariant and
equivariant representations** across two outputs. The trick: add an
**autoregressive memory module** that aggregates a _sequence_ of
action-conditioned observations into a global representation. Empirically:

- The **encoder output** (per-view) is **action-equivariant** — it transforms
  with the action.
- The **autoregressor output** (aggregated) is **action-invariant** — stable
  across actions. This resolves the invariance↔equivariance trade-off that
  plagues single-representation JEPAs like IWM, where a bigger predictor buys
  equivariance at the cost of invariant-linear-probe performance.

## Problem & motivation

- Action-conditioned JEPAs (IWM, I-JEPA) have a single prediction loss, and the
  size of the predictor controls a trade-off: **larger predictor → more
  equivariant world model → worse invariant linear probe** (Garrido et al.
  2024). The same representational space is forced to hold both invariant and
  equivariant features, creating the trade-off.
- Animals learn by _sequences of active interactions_ (saccades, manipulation),
  not single view-pairs. seq-JEPA brings this sequential structure into JEPA.

## Method

- **Encoder** $E_\theta$: maps each observation $x_t$ → latent
  $z_t = E_\theta(x_t)$.
- **Autoregressive memory** $A_\phi$: aggregates $\{z_1, \dots, z_t\}$ + actions
  $\{a_1, \dots, a_{t-1}\}$ → a global representation
  $h_t = A_\phi(z_{1:t}, a_{1:t-1})$.
- **Predictor** $P_\psi$: takes $(h_t, a_t)$ → predicts $\hat{z}_{t+1}$,
  compared to the EMA-target encoder output $z_{t+1}$ via latent L1/L2.
- Two evaluation settings:
  1. **Predictive learning across saccades** — low-res visual patches sampled
     from image saliency, no hand-crafted augmentations; learns SSL image reps
     from active perception.
  2. **Invariance-equivariance trade-off** — measures both invariant and
     equivariant benchmarks; shows seq-JEPA gets both, where IWM gets one at the
     expense of the other.

## Key result

- **Emergent disentanglement**: encoder output ≡ equivariant, autoregressor
  output ≡ invariant. No explicit equivariance loss needed — the architecture
  induces it.
- Competitive with both invariant SSL (for linear probes) and equivariant SSL
  (for fine-grained / segmentation tasks) on standard benchmarks.

## Relevance to the EB-JEPA hackathon

Two transferable ideas for an action-conditioned point-cloud track:

1. **Resolving the invariance↔equivariance trade-off for point clouds.** The
   `examples/pointcloud/` README studies _view-invariance_ (rotation:
   none→z→SO(3)) via two-view VICReg. If you make it action-conditioned (e.g.
   action = rotation/scale parameters, à la IWM), you hit exactly the trade-off
   seq-JEPA solves: the encoder should be equivariant to the action (so the
   predictor can predict the transformed latent), but the global feature should
   be invariant (for the linear probe). seq-JEPA's two-output design gives you
   both.
2. **Autoregressive memory for point-cloud sequences.** If the point-cloud track
   is extended to spatio-temporal data (e.g. STRL-style moving LiDAR),
   seq-JEPA's autoregressor over action-conditioned point-cloud latents is the
   natural predictor — and it splits the invariant global feature (for
   classification) from the equivariant per-frame feature (for scene-flow
   prediction) for free.
3. **Action as the "rescaling" parameter.** For "rescaling point cloud based on
   an action," IWM and seq-JEPA are the recipe: condition the predictor on the
   scale/rotation parameters, predict the latent of the transformed cloud.
   seq-JEPA adds that this doesn't collapse the invariant probe — important if
   you want to keep the downstream classification probe working.

## Caveats / open threads

- Image-domain only; no point-cloud experiments in the paper.
- The autoregressive memory adds a transformer over the sequence — modest extra
  compute.
- The disentanglement is _emergent_ (empirical), not theoretically guaranteed.
