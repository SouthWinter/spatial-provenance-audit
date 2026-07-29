# SCOPE Pure-Coverage Ablation

Differences are pure coverage (alpha=0) minus saliency plus coverage (alpha=1).
Confidence intervals use paired image-cluster bootstrap.

| Split | Variant | Acc. | hFPR | PosECR | NegSRC | ECE15 | Brier | Correct-positive low/zero ECR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| development | saliency plus coverage | 0.598 | 0.508 | 0.642 | 0.642 | 0.036 | 0.237 | 32.4% / 15.1% |
| development | pure coverage | 0.587 | 0.630 | 0.623 | 0.623 | 0.040 | 0.236 | 35.8% / 11.9% |
| confirmation | saliency plus coverage | 0.586 | 0.562 | 0.614 | 0.614 | 0.037 | 0.237 | 34.3% / 16.9% |
| confirmation | pure coverage | 0.579 | 0.648 | 0.619 | 0.619 | 0.050 | 0.236 | 35.5% / 15.6% |

| Confirmation difference | Estimate | 95% CI |
|---|---:|---:|
| accuracy | -0.007 | [-0.024, +0.010] |
| hfpr | +0.086 | [+0.052, +0.120] |
| positive ecr | +0.005 | [-0.038, +0.047] |
| negative source coverage | +0.005 | [-0.038, +0.046] |
