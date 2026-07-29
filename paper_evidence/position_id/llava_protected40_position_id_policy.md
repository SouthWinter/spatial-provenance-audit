# LLaVA-1.5-7B Protected (40%) main-mask position-ID policy audit

Compact uses the model's ordinary physically shortened-prefix path, which assigns consecutive logical positions; preserve explicitly keeps the pre-pruning logical position IDs. Both policies use contiguous physical cache slots.

| Policy | Raw all Acc. | Raw all hFPR | Own-dev test Acc. | Own-dev test hFPR | Compact-shared test Acc. | Compact-shared test hFPR | Test AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Compact | 0.661 | 0.298 | 0.655 | 0.272 | 0.655 | 0.272 | 0.702 |
| Preserve | 0.654 | 0.362 | 0.647 | 0.280 | 0.646 | 0.299 | 0.694 |

Development thresholds: compact=0.020278, preserve=0.039729. Development/test sizes: 464/536.

At the compact-derived shared threshold, 155 of 536 held-out predictions flip; mean absolute margin change is 0.1759 and the maximum is 0.8078.

Mask invariance: kept-index mismatches=0, ECR mismatches=0, sample-ID mismatches=0.

Interpretation: position handling is an inference-policy variable, not a selector or coverage change. Answer metrics must therefore be reported with the position policy fixed and named.
