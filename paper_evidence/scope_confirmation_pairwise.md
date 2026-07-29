# Paired pruning statistics

Bootstrap samples: 10000
Seed: 20260721

| comparison | acc left | acc right | acc diff | acc 95% CI | McNemar p | hFPR left | hFPR right | hFPR diff | hFPR 95% CI | hFPR McNemar p |
|---|---:|---:|---:|---|---:|---:|---:|---:|---|---:|
| scope_vs_full | 0.5860 | 0.6250 | -0.0390 | [-0.0710, -0.0070] | 0.0176 | 0.5620 | 0.3160 | +0.2460 | [+0.2020, +0.2901] | 4.35e-26 |
| scope_vs_target | 0.5860 | 0.6240 | -0.0380 | [-0.0680, -0.0080] | 0.01452 | 0.5620 | 0.5740 | -0.0120 | [-0.0573, +0.0331] | 0.6683 |
| protected_vs_scope | 0.6430 | 0.5860 | +0.0570 | [+0.0250, +0.0890] | 0.0006425 | 0.3520 | 0.5620 | -0.2100 | [-0.2561, -0.1641] | 4.157e-17 |
| random_vs_scope | 0.6690 | 0.5860 | +0.0830 | [+0.0500, +0.1170] | 1.201e-06 | 0.2560 | 0.5620 | -0.3060 | [-0.3512, -0.2615] | 5.631e-35 |
| scope_vs_visionzip | 0.5860 | 0.5820 | +0.0040 | [-0.0230, +0.0310] | 0.826 | 0.5620 | 0.6700 | -0.1080 | [-0.1496, -0.0681] | 3.291e-07 |

Positive diff means the left run is higher than the right run. Lower hFPR is better.
McNemar b/c counts are available in the CSV for exact discordance auditing.
