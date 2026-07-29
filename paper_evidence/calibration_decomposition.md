# Qwen/LLaVA Confirmation Calibration Decomposition

A single threshold is selected from each backbone's independent Full-prefix development scores and then applied unchanged to every locked-confirmation selector. AUROC is threshold-free. The primary paper still uses the frozen zero threshold for these backbones; this analysis is diagnostic.

| Model | Method | n | AUROC | Delta vs. Full | t=0 Acc. | t=0 hFPR | Full-dev t | Shared Acc. | Shared hFPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-VL-8B | Full | 1000 | 0.856 | +0.000 | 0.783 | 0.248 | 1.062 | 0.812 | 0.132 |
| Qwen3-VL-8B | Target (30%) | 1000 | 0.855 | +0.000 | 0.786 | 0.240 | 1.062 | 0.810 | 0.142 |
| Qwen3-VL-8B | Random (30%) | 1000 | 0.776 | -0.080 | 0.739 | 0.160 | 1.062 | 0.757 | 0.072 |
| Qwen3-VL-8B | Grid (30%) | 1000 | 0.755 | -0.100 | 0.723 | 0.142 | 1.062 | 0.723 | 0.080 |
| Qwen3-VL-8B | VisionZip (30%) | 1000 | 0.788 | -0.068 | 0.748 | 0.198 | 1.062 | 0.764 | 0.110 |
| LLaVA-1.5-7B | Full | 1000 | 0.666 | +0.000 | 0.625 | 0.316 | 0.088 | 0.623 | 0.220 |
| LLaVA-1.5-7B | Protected (40%) | 1000 | 0.694 | +0.028 | 0.643 | 0.352 | 0.088 | 0.653 | 0.224 |
| LLaVA-1.5-7B | Random (40%) | 1000 | 0.712 | +0.046 | 0.669 | 0.256 | 0.088 | 0.657 | 0.150 |
| LLaVA-1.5-7B | Target (40%) | 1000 | 0.713 | +0.047 | 0.624 | 0.574 | 0.088 | 0.653 | 0.442 |
| LLaVA-1.5-7B | SCOPE (40%) | 1000 | 0.644 | -0.022 | 0.586 | 0.562 | 0.088 | 0.593 | 0.454 |
| LLaVA-1.5-7B | AnchorPrune (40%) | 1000 | 0.642 | -0.024 | 0.578 | 0.586 | 0.088 | 0.596 | 0.474 |
| LLaVA-1.5-7B | CoIn (40%) | 1000 | 0.664 | -0.002 | 0.579 | 0.674 | 0.088 | 0.595 | 0.562 |

Qwen Target matches Full's threshold-free ranking (0.855 versus 0.856), whereas Random, Grid, and VisionZip are lower. LLaVA Target has higher AUROC than Full (0.713 versus 0.666), but its shared-threshold hFPR remains 0.442. The LLaVA result therefore contains a substantial calibration shift without reducing the comparison to calibration alone.
