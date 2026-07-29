# ECR Construct-Validity Audit

This is a cached, positive-probe-only analysis. ECR measures retained spatial provenance; it is not treated as causal use. Local provenance precision (LPP) is the fraction of the union area of retained cells touching evidence that lies inside the evidence region. Geo-F1 is the harmonic mean of ECR and LPP.

## Geometry Summary

| Model | n | ECR | LPP | Geo-F1 | Occlusion n | Deletion n |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-VL-8B | 500 | 0.651 | 0.205 | 0.295 | 100 | 500 |
| LLaVA-1.5-7B | 500 | 0.458 | 0.112 | 0.171 | 50 | 0 |
| InternVL3.5-8B | 500 | 0.916 | 0.166 | 0.255 | 50 | 500 |

## ECR Association With Interventions

Spearman correlations are primary because ECR is bounded and often tied. Intervals are percentile bootstrap intervals over paired samples.

| Model | Intervention | n | Spearman r (95% CI) |
| --- | --- | ---: | ---: |
| Qwen3-VL-8B | occlusion_drop | 100 | 0.317 [0.122, 0.495] |
| Qwen3-VL-8B | occlusion_specific_drop | 100 | 0.334 [0.138, 0.511] |
| Qwen3-VL-8B | deletion_drop | 500 | 0.308 [0.227, 0.389] |
| Qwen3-VL-8B | evidence_restoration_gain | 500 | 0.308 [0.226, 0.389] |
| Qwen3-VL-8B | restoration_specific_gain | 500 | 0.289 [0.204, 0.373] |
| LLaVA-1.5-7B | occlusion_drop | 50 | 0.129 [-0.159, 0.387] |
| LLaVA-1.5-7B | occlusion_specific_drop | 50 | 0.270 [-0.012, 0.519] |
| InternVL3.5-8B | occlusion_drop | 50 | 0.109 [-0.203, 0.396] |
| InternVL3.5-8B | occlusion_specific_drop | 50 | 0.158 [-0.147, 0.434] |
| InternVL3.5-8B | deletion_drop | 500 | 0.029 [-0.058, 0.112] |
| InternVL3.5-8B | evidence_restoration_gain | 500 | 0.029 [-0.056, 0.114] |
| InternVL3.5-8B | restoration_specific_gain | 500 | 0.035 [-0.052, 0.121] |

## Conditional Association

Partial Spearman correlations residualize ranked ECR and ranked intervention effects against log evidence-box area, log median token-cell area, and the Full-prefix yes margin. Constant controls within a model are dropped automatically. These are construct-validity diagnostics, not causal estimates.

| Model | Intervention | n | Partial Spearman r (95% CI) |
| --- | --- | ---: | ---: |
| Qwen3-VL-8B | occlusion_specific_drop | 100 | 0.102 [-0.082, 0.285] |
| Qwen3-VL-8B | deletion_drop | 500 | 0.288 [0.203, 0.371] |
| LLaVA-1.5-7B | occlusion_specific_drop | 50 | 0.275 [-0.025, 0.535] |
| InternVL3.5-8B | occlusion_specific_drop | 50 | 0.159 [-0.119, 0.414] |
| InternVL3.5-8B | deletion_drop | 500 | 0.011 [-0.075, 0.096] |

## Interpretation Guardrails

- A positive association supports convergent validity: masks with more annotated-region provenance tend to lose more target support when that region is removed or occluded.
- A null association does not prove ECR invalid. It can arise from restricted ECR range, contextualized receptive fields, weak model use of OCR evidence, or low intervention power.
- LPP and Geo-F1 are geometry diagnostics, not additional selector objectives. They prevent a coarse token cell from receiving the same interpretation as a tightly localized cell solely because both cover the box.
- InternVL thumbnail cells overlap tiled cells by design; union-area calculations avoid double counting. Qwen grids are reconstructed after spatial merging.
- Full sample rows, scale strata, Pearson correlations, and valid bootstrap counts are in the companion CSV files.
