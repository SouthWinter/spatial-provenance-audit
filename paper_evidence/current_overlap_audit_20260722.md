# Current Overlap Audit (2026-07-22)

## Calibration and Evidence Coverage

The 2026-07-19 v2 of *When Does Visual Token Pruning Improve Calibration? The
Role of Evidence Coverage in MLLMs* (arXiv:2604.12035) studies confidence
calibration under visual-token pruning. Its controlled selector axis is SCOPE
with saliency exponent alpha in {0, 0.5, 1}; alpha=0 is pure facility-location
feature coverage and alpha=1 is the public default saliency-weighted coverage.
Its primary evidence is POPE first-token confidence, ECE, and related
calibration metrics.

Our added LLaVA-1.5-7B audit evaluates the same alpha=0 and alpha=1 selector
endpoints at 40% retention on the locked TextOCR-Hard confirmation set. It uses
physical token removal, compact logical positions, and the same SCOPE code path
for both endpoints. Pure coverage minus default SCOPE gives:

- accuracy: -0.007, image-cluster 95% CI [-0.024, +0.010];
- hFPR: +0.086, image-cluster 95% CI [+0.052, +0.120], exact McNemar
  p=8.91e-7;
- PosECR: +0.005, image-cluster 95% CI [-0.038, +0.047];
- sequence-likelihood ECE15: 0.050 versus 0.037;
- sequence-likelihood Brier: 0.236 versus 0.237.

The confidence readout differs from the concurrent paper, so the ECE values are
not a cross-paper replication. The controlled endpoint comparison instead
shows that feature coverage, confidence calibration, annotation-defined
spatial provenance, and hard-negative answer risk are distinct evaluation
targets.

Primary source: https://arxiv.org/abs/2604.12035

## TOPS

*TOPS: First-Principles Visual Token Pruning via Constructing Token Optimal
Preservation Sets for Efficient MLLM Inference* (arXiv:2606.27161, submitted
2026-06-25) combines task relevance, feature coverage, and semantic diversity
in a two-stage, multi-layer pruning procedure and reports results on LLaVA,
Qwen2.5-VL, and InternVL3.

The public arXiv source inspected on 2026-07-22 links the base LLaVA,
LLaVA-NeXT, lmms-eval, and VLMEvalKit repositories, but does not link a TOPS
method repository, release tag, or immutable implementation revision. We cite
the method as related work but do not label an independently reconstructed
multi-stage implementation as an official baseline. The supplement records
this implementation boundary explicitly.

Primary source: https://arxiv.org/abs/2606.27161

## Revision Decision

1. Cite and distinguish the calibration paper in Related Work.
2. Report the matched SCOPE alpha endpoint audit in the main results and full
   development/confirmation statistics in the supplement.
3. Keep TOPS as cited related work until an attributable implementation is
   available; do not present an unverifiable port as code-parity evidence.
