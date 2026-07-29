# CoIn TextOCR-Hard Audit: Development

CoIn is a paper-algorithm port of CVPR 2026 Algorithm 1, not an official-code reproduction. The port uses projected LLaVA tokens, mean non-image prompt alignment, and the reported LLaVA-1.5 128-token setting alpha=0.9, beta=0.6 without TextOCR tuning.

| method | n | keep | accuracy | hFPR | PosECR | NegSRC | selector ms | overhead ms | forward ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full | 1000 | 1.000 | 0.626 | 0.304 | 1.000 | 1.000 | 0.0 | 0.0 | 0.0 |
| Protected (40%) | 1000 | 0.401 | 0.661 | 0.298 | 1.000 | 1.000 | 1.0 | 2.5 | 47.4 |
| Target (40%) | 1000 | 0.401 | 0.623 | 0.496 | 0.458 | 0.439 | 0.2 | 5.5 | 50.2 |
| CoIn (40%) | 1000 | 0.401 | 0.595 | 0.632 | 0.692 | 0.686 | 578.5 | 592.6 | 698.4 |
| SCOPE (40%) | 1000 | 0.401 | 0.598 | 0.508 | 0.642 | 0.642 | 129.0 | 130.0 | 184.5 |
| Random (40%) | 1000 | 0.401 | 0.660 | 0.256 | 0.402 | 0.437 | 0.1 | 1.0 | 44.1 |
| Grid (40%) | 1000 | 0.401 | 0.649 | 0.258 | 0.391 | 0.391 | 0.0 | 0.0 | 75.1 |
| VisionZip (40%) | 1000 | 0.401 | 0.580 | 0.650 | 0.575 | 0.575 | 12.9 | 13.1 | 52.3 |

| comparison | delta accuracy [image-cluster 95% CI] | p | delta hFPR [image-cluster 95% CI] | p |
|---|---:|---:|---:|---:|
| CoIn (40%) - Full | -0.031 [-0.057, -0.007] | 0.08361 | +0.328 [+0.288, +0.370] | 3.571e-48 |
| CoIn (40%) - Protected (40%) | -0.066 [-0.092, -0.041] | 0.0002221 | +0.334 [+0.288, +0.380] | 9.396e-39 |
| CoIn (40%) - Target (40%) | -0.028 [-0.053, -0.003] | 0.0672 | +0.136 [+0.090, +0.182] | 1.631e-08 |
| CoIn (40%) - SCOPE (40%) | -0.003 [-0.022, +0.015] | 0.8634 | +0.124 [+0.094, +0.156] | 6.388e-15 |
| CoIn (40%) - Random (40%) | -0.065 [-0.091, -0.040] | 0.0003856 | +0.376 [+0.332, +0.420] | 9.694e-53 |
| CoIn (40%) - Grid (40%) | -0.054 [-0.079, -0.030] | 0.004314 | +0.374 [+0.330, +0.418] | 2.053e-49 |
| CoIn (40%) - VisionZip (40%) | +0.015 [-0.005, +0.034] | 0.2668 | -0.018 [-0.058, +0.022] | 0.4394 |


# CoIn TextOCR-Hard Audit: Locked confirmation

CoIn is a paper-algorithm port of CVPR 2026 Algorithm 1, not an official-code reproduction. The port uses projected LLaVA tokens, mean non-image prompt alignment, and the reported LLaVA-1.5 128-token setting alpha=0.9, beta=0.6 without TextOCR tuning.

| method | n | keep | accuracy | hFPR | PosECR | NegSRC | selector ms | overhead ms | forward ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full | 1000 | 1.000 | 0.625 | 0.316 | 1.000 | 1.000 | 0.0 | 0.0 | 0.0 |
| Protected (40%) | 1000 | 0.401 | 0.643 | 0.352 | 1.000 | 1.000 | 1.0 | 11.2 | 90.1 |
| Target (40%) | 1000 | 0.401 | 0.624 | 0.574 | 0.471 | 0.443 | 0.3 | 5.7 | 56.1 |
| CoIn (40%) | 1000 | 0.401 | 0.579 | 0.674 | 0.666 | 0.680 | 579.9 | 594.4 | 697.8 |
| SCOPE (40%) | 1000 | 0.401 | 0.586 | 0.562 | 0.614 | 0.614 | 39.0 | 39.3 | 89.2 |
| Random (40%) | 1000 | 0.401 | 0.669 | 0.256 | 0.372 | 0.389 | 0.2 | 2.5 | 69.8 |
| VisionZip (40%) | 1000 | 0.401 | 0.582 | 0.670 | 0.563 | 0.563 | 17.2 | 17.6 | 71.4 |

| comparison | delta accuracy [image-cluster 95% CI] | p | delta hFPR [image-cluster 95% CI] | p |
|---|---:|---:|---:|---:|
| CoIn (40%) - Full | -0.046 [-0.071, -0.021] | 0.01426 | +0.358 [+0.314, +0.402] | 1.864e-45 |
| CoIn (40%) - Protected (40%) | -0.064 [-0.090, -0.038] | 0.0004118 | +0.322 [+0.276, +0.368] | 1.548e-34 |
| CoIn (40%) - Target (40%) | -0.045 [-0.071, -0.019] | 0.002044 | +0.100 [+0.054, +0.146] | 3.756e-05 |
| CoIn (40%) - SCOPE (40%) | -0.007 [-0.026, +0.013] | 0.6322 | +0.112 [+0.076, +0.148] | 3.101e-09 |
| CoIn (40%) - Random (40%) | -0.090 [-0.117, -0.063] | 1.857e-06 | +0.418 [+0.372, +0.462] | 8.694e-58 |
| CoIn (40%) - VisionZip (40%) | -0.003 [-0.022, +0.016] | 0.8785 | +0.004 [-0.036, +0.044] | 0.9227 |
