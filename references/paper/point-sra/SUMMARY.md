# Point-SRA — Self-Representation Alignment for 3D Representation Learning

**Authors:** Lintong Wei, Jian Lu (Xi'an Polytechnic Univ.), Haozhe Cheng, Jihua
Zhu (Xi'an Jiaotong Univ.), Kaibing Zhang (Xi'an Polytechnic Univ.)
**Venue/Year:** AAAI 2026 (Vol. 40, No. 13) **arXiv:**
[2601.01746](https://arxiv.org/abs/2601.01746) · **AAAI:**
[ojs.aaai.org/38026](https://ojs.aaai.org/index.php/AAAI/article/view/38026)

## TL;DR

A **post-MAE** self-supervised 3D representation learner that fixes two flaws of
Point-MAE-style methods: (i) the fixed-mask-ratio assumption (which ignores that
_low_ ratios preserve geometry and _high_ ratios encourage semantics — they are
complementary) and (ii) point-wise deterministic reconstruction (which ignores
the inherent uncertainty / multi-solution nature of 3D completion). Point-SRA
replaces point-wise reconstruction with **MeanFlow probabilistic
reconstruction**, runs the MAE at **multiple mask ratios simultaneously** and
aligns their representations via a **Dual Self-Representation Alignment (Dual
SRA)** mechanism, then transfers the learned point-cloud distribution to
downstream tasks via a **Flow-Conditioned Fine-Tuning (FCFT) Architecture**.
+5.37 % over Point-MAE on ScanObjectNN; +5.12 % over MaskPoint on ScanNetV2 3D
detection.

## Problem & motivation

- MAE-style 3D SSL fixes a single masking ratio and reconstructs masked points
  deterministically.
- The paper proves (entropy / mutual-information argument) that different mask
  ratios yield **complementary** subspaces: a low ratio $\to$ geometric detail
  projection $\pi_{geo}$, a high ratio $\to$ semantic projection $\pi_{sem}$,
  with $\mathcal{C}(Z)=\mathcal{H}(Z)/\mathcal{I}(\mathcal{P}_{semantic};Z)$.
- 3D reconstruction is inherently uncertain (many plausible completions), so
  point-wise L1/Chamfer targets are over-constrained.

## Method — three pillars

### 1. MeanFlow Transformer (MFT) for probabilistic reconstruction

- A continuous trajectory $\{z_t\}_{t\in[0,1]}$ with $z_0$ = target point cloud,
  $z_1\sim\mathcal{N}(0,I)$, linear interpolation $z_t=(1-t)z_0 + t z_1$.
- **Cross-modal conditional embeddings**: image features $f_{img}$ + text
  features $f_{text}$ (from dedicated image/text encoders) condition the flow,
  giving diverse probabilistic reconstructions instead of a single point-wise
  target.
- MeanFlow's trajectory-based learning naturally aligns representations across
  temporal states.

### 2. Dual Self-Representation Alignment (Dual SRA)

Two internal self-distillation paths:

- **MAE-SRA** — aligns representations between MAE branches with _different mask
  ratios_ (geometry ↔ semantics knowledge transfer).
- **MFT-SRA** — temporal alignment: pick two time steps $t_a>t_b$; the student
  MFT processes $z_{t_a}$ at $t_a$, the EMA-teacher MFT processes $z_{t_b}$ at
  $t_b$:
  $$h_{t_a}=F_{MF}(z_{t_a},t_a,c),\quad h_{t_b}=F_{MF}^{EMA}(z_{t_b},t_b,c)$$
  and align them — representations at different time steps are also
  complementary.

### 3. Flow-Conditioned Fine-Tuning (FCFT) Architecture

At fine-tuning, the **pre-trained MFT is frozen** and used to compute a _flow
vector_ for each point-cloud group from geometry alone (no image/text at FT
time):
$$F_{u_\theta}=MFT_{frozen}(Center, t, r),\quad Center\in\mathbb{R}^{G\times 3}$$
with $t, r$ sampled as in pretraining. The flow vector is fused into the
downstream head, carrying the geometric-distribution knowledge learned during
SSL into detection/segmentation.

### Total loss

$$\mathcal{L}_{total}=\mathcal{L}_{recon}+\lambda_{flow}\mathcal{L}_{MFM}+\mathcal{L}_{CSC}+\lambda_{mae\text{-}sra}\mathcal{L}_{mae\text{-}sra}+\lambda_{mft\text{-}sra}\mathcal{L}_{mft\text{-}sra}$$
with $\lambda_{flow}=0.5$,
$\lambda_{mae\text{-}sra}=\lambda_{mft\text{-}sra}=0.2$.

## Key results

| Task                       | Metric        | Point-SRA   | Baseline  | Δ           |
| -------------------------- | ------------- | ----------- | --------- | ----------- |
| ScanObjectNN (PB_T50_RS)   | OA            | —           | Point-MAE | **+5.37 %** |
| 3D detection (ScanNetV2)   | AP@50         | **47.3 %**  | MaskPoint | +5.12 %     |
| Intracranial aneurysm seg. | mIoU artery   | **96.07 %** | —         | —           |
| Intracranial aneurysm seg. | mIoU aneurysm | **86.87 %** | —         | —           |

Ablation: MeanFlow alone gives **+5.45 %** on PB_T50_RS vs point-wise
reconstruction, validating the probabilistic-reconstruction thesis.

## Relevance to the EB-JEPA hackathon

Point-SRA is **not a JEPA** — it is a generative/probabilistic MAE descendant.
But two ideas are directly portable to a JEPA-on-point-cloud track
(`examples/pointcloud/`):

1. **Multi-mask-ratio complementarity → multi-target JEPA.** Point-SRA's proof
   that low vs high mask ratios span complementary subspaces is exactly the
   motivation for Point-JEPA's "multiple target blocks" and for EB-JEPA's
   multi-target prediction. A concrete upgrade to the current two-view VICReg
   track: sample target blocks at _different_ mask ratios and predict them
   jointly.
2. **Self-distillation across time steps → EMA-teacher JEPA.** MFT-SRA aligning
   $h_{t_a}$ (student) to $h_{t_b}$ (EMA teacher) is structurally the same as a
   JEPA's EMA target encoder + stop-grad, but applied along a _flow trajectory_
   rather than a mask. This is a concrete bridge between diffusion/flow training
   and JEPA's latent-prediction objective — relevant if you want to combine the
   EB-JEPA predictor with a flow-matching prior.
3. **Frozen-pretrained-encoder fine-tuning.** The FCFT pattern (freeze the SSL
   encoder, inject its features into a downstream head) is exactly the
   `eval.py:probe` pattern the pointcloud track asks for — Point-SRA shows it
   generalizes from classification to detection/segmentation.

## Caveats / open threads

- Relies on **cross-modal image+text features** during pretraining — heavier
  infrastructure than a pure point-cloud SSL method; the FCFT stage drops them,
  but pretraining needs them.
- Code release not confirmed at time of writing; check the arXiv/AAAI page.
- MeanFlow adds a generative component — sits closer to diffusion/flow models
  than to JEPA's non-generative latent prediction; the relationship to EB-JEPA's
  energy-based formulation is an open theoretical question.
- "Self-Representation Alignment" is a reused name — note the unrelated ICLR
  2026 _SRA for Diffusion Transformers_ (Diffusion-DiT self-alignment,
  `github.com/vvvvvjdy/SRA`); don't confuse the two.
