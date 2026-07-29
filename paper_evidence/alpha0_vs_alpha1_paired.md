# Paired pruning statistics

Bootstrap samples: 10000
Seed: 20260722

| comparison | acc left | acc right | acc diff | acc 95% CI | McNemar p | hFPR left | hFPR right | hFPR diff | hFPR 95% CI | hFPR McNemar p |
|---|---:|---:|---:|---|---:|---:|---:|---:|---|---:|
| alpha0-vs-alpha1 | 0.5790 | 0.5860 | -0.0070 | [-0.0300, +0.0160] | 0.611 | 0.6480 | 0.5620 | +0.0860 | [+0.0522, +0.1194] | 8.91e-07 |

Positive diff means the left run is higher than the right run. Lower hFPR is better.
McNemar b/c counts are available in the CSV for exact discordance auditing.
