# InternVL3.5-8B Soft evidence (50%) main-mask position-ID policy audit

Compact uses the model's ordinary physically shortened-prefix path, which assigns consecutive logical positions; preserve explicitly keeps the pre-pruning logical position IDs. Both policies use contiguous physical cache slots.

| Policy | Raw all Acc. | Raw all hFPR | Own-dev test Acc. | Own-dev test hFPR | Compact-shared test Acc. | Compact-shared test hFPR | Test AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Compact | 0.500 | 1.000 | 0.653 | 0.328 | 0.653 | 0.328 | 0.696 |
| Preserve | 0.502 | 0.996 | 0.647 | 0.343 | 0.647 | 0.272 | 0.678 |

Development thresholds: compact=2.344476, preserve=2.218685. Development/test sizes: 464/536.

At the compact-derived shared threshold, 101 of 536 held-out predictions flip; mean absolute margin change is 0.3425 and the maximum is 1.3021.

Mask invariance: kept-index mismatches=0, ECR mismatches=0, sample-ID mismatches=0.

Interpretation: position handling is an inference-policy variable, not a selector or coverage change. Answer metrics must therefore be reported with the position policy fixed and named.
