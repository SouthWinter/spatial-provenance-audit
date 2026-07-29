# Locked-Confirmation Random-Seed Audit

Six fixed random masks are evaluated on the same 500-image locked confirmation split. The two-stage percentile bootstrap first resamples random-mask seeds and then image clusters.

| Random control | Accuracy | hFPR | PosECR |
| --- | ---: | ---: | ---: |
| qwen3_8b_random_0p30 | 0.739 | 0.160 | 0.270 |
| qwen3_8b_random_0p30_seed101 | 0.726 | 0.164 | 0.294 |
| qwen3_8b_random_0p30_seed202 | 0.733 | 0.168 | 0.279 |
| qwen3_8b_random_0p30_seed303 | 0.736 | 0.158 | 0.310 |
| qwen3_8b_random_0p30_seed404 | 0.738 | 0.162 | 0.315 |
| qwen3_8b_random_0p30_seed505 | 0.727 | 0.168 | 0.298 |

| Target minus Random | Difference (95% CI) | Seeds | Images |
| --- | ---: | ---: | ---: |
| accuracy | +0.053 [+0.033, +0.073] | 6 | 500 |
| PosECR | +0.325 [+0.286, +0.365] | 6 | 500 |

The target selector is deterministic; seed resampling quantifies uncertainty from the matched-budget random-mask control, while image resampling preserves positive/negative pairing.
