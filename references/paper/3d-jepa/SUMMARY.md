# 3D-JEPA — A Joint Embedding Predictive Architecture for 3D Self-Supervised Representation Learning

**Authors:** (see arXiv listing) **Venue/Year:** arXiv 2024 **arXiv:**
[2409.15803](https://arxiv.org/abs/2409.15803)

## TL;DR

Concurrent work to Point-JEPA. Claims to be the **first non-generative**
point-cloud pretraining architecture. Uses a **multi-block sampling** strategy
to draw one context block + several semantically-rich target blocks from the
same cloud (no hand-crafted augmentations), then predicts the target-block
representations from the context block in feature space. Avoids both (i)
augmentation bias of invariance methods and (ii) the over-focus on irrelevant
detail of generative methods (e.g. Point-MAE reconstructing regular patterns).

## Method

- Multi-block sampling: single context + multiple targets from the same point
  cloud.
- Encoder predicts high-level concepts of target blocks from context in feature
  space.
- Argues point clouds have highly structured information (planes look like
  planes) so generative reconstruction wastes capacity on the predictable
  "regular pattern"; JEPA sidesteps this.

## Relevance to the EB-JEPA hackathon

- Alternative patch-sampling story to Point-JEPA's greedy sequencer: instead of
  _ordering_ patches and masking contiguous runs, **sample multiple disjoint
  target blocks** directly.
- Useful comparison point when designing the context/target split for the
  `examples/pointcloud/` track — multi-block sampling may be simpler to
  implement on top of an FPS-KNN tokenizer than the greedy sequencer.

## Caveats / open threads

- Less widely cited / less mature tooling than Point-JEPA.
- Detailed results (ModelNet40 numbers, code release) need to be confirmed from
  the latest arXiv version before relying on them.
