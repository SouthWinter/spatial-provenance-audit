# Locked-Confirmation Human-QC Audit

All 500 locked-confirmation hard negatives received final human QC. A frozen 100-row subset was independently labeled before adjudication.

## Label Quality

- Strictly valid near-miss probes: 465/500.
- Target absence confirmed after normalization: 465/500.
- Source not visually identifiable: 34/500.
- Target present in the same image: 1/500.
- Frozen secondary subset: 0.900 binary-validity agreement, $\kappa=0.445$; 10 rows adjudicated.

## hFPR and Paired-Accuracy Sensitivity

| Model | Method | Original hFPR | Human-valid hFPR | $\Delta$hFPR | Original acc. | Human-valid acc. | $\Delta$acc. |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3-VL-8B | Full | 0.248 | 0.249 | +0.001 | 0.783 | 0.801 | +0.018 |
| Qwen3-VL-8B | Target (30%) | 0.240 | 0.245 | +0.005 | 0.786 | 0.798 | +0.012 |
| Qwen3-VL-8B | Random (30%) | 0.160 | 0.161 | +0.001 | 0.739 | 0.752 | +0.013 |
| Qwen3-VL-8B | Grid (30%) | 0.142 | 0.144 | +0.002 | 0.723 | 0.735 | +0.012 |
| Qwen3-VL-8B | VisionZip (30%) | 0.198 | 0.202 | +0.004 | 0.748 | 0.763 | +0.015 |
| LLaVA-1.5-7B | Full | 0.316 | 0.320 | +0.004 | 0.625 | 0.631 | +0.006 |
| LLaVA-1.5-7B | Protected (40%) | 0.352 | 0.353 | +0.001 | 0.643 | 0.651 | +0.008 |
| LLaVA-1.5-7B | Random (40%) | 0.256 | 0.260 | +0.004 | 0.669 | 0.674 | +0.005 |
| LLaVA-1.5-7B | Target (40%) | 0.574 | 0.583 | +0.009 | 0.624 | 0.625 | +0.001 |
| LLaVA-1.5-7B | SCOPE (40%) | 0.562 | 0.561 | -0.001 | 0.586 | 0.590 | +0.004 |
| LLaVA-1.5-7B | AnchorPrune (40%) | 0.586 | 0.583 | -0.003 | 0.578 | 0.584 | +0.006 |
| LLaVA-1.5-7B | CoIn (40%) | 0.674 | 0.669 | -0.005 | 0.579 | 0.584 | +0.005 |
| LLaVA-1.5-7B | VisionZip (40%) | 0.670 | 0.669 | -0.001 | 0.582 | 0.586 | +0.004 |

The original 500-image locked table remains the prespecified primary readout. The 465-image human-valid subset is a post-QC sensitivity analysis.

## Complete Valid-465 Readout

| Model | Method | Accuracy | hFPR | AUROC | PosECR | AncECR | NegSRC |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3-VL-8B | Full | 0.801 | 0.249 | 0.877 | 1.000 | 1.000 | 1.000 |
| Qwen3-VL-8B | Target (30%) | 0.798 | 0.245 | 0.875 | 0.634 | 0.634 | 0.514 |
| Qwen3-VL-8B | Random (30%) | 0.752 | 0.161 | 0.792 | 0.270 | 0.270 | 0.298 |
| Qwen3-VL-8B | Grid (30%) | 0.735 | 0.144 | 0.765 | 0.321 | 0.321 | 0.321 |
| Qwen3-VL-8B | VisionZip (30%) | 0.763 | 0.202 | 0.807 | 1.000 | 0.851 | 1.000 |
| LLaVA-1.5-7B | Full | 0.631 | 0.320 | 0.675 | 1.000 | 1.000 | 1.000 |
| LLaVA-1.5-7B | Protected (40%) | 0.651 | 0.353 | 0.704 | 1.000 | 1.000 | 1.000 |
| LLaVA-1.5-7B | Random (40%) | 0.674 | 0.260 | 0.719 | 0.375 | 0.375 | 0.388 |
| LLaVA-1.5-7B | Target (40%) | 0.625 | 0.583 | 0.720 | 0.479 | 0.479 | 0.448 |
| LLaVA-1.5-7B | SCOPE (40%) | 0.590 | 0.561 | 0.652 | 0.613 | 0.613 | 0.613 |
| LLaVA-1.5-7B | AnchorPrune (40%) | 0.584 | 0.583 | 0.650 | 0.636 | 0.636 | 0.641 |
| LLaVA-1.5-7B | CoIn (40%) | 0.584 | 0.669 | 0.672 | 0.672 | 0.672 | 0.688 |
| LLaVA-1.5-7B | VisionZip (40%) | 0.586 | 0.669 | 0.672 | 1.000 | 0.562 | 1.000 |

## Primary Paired Comparison

- all_locked_negatives: $n=500$, Target--Full accuracy +0.003 [-0.014,+0.020].
- target_absence_confirmed: $n=465$, Target--Full accuracy -0.003 [-0.022,+0.015].
- strict_valid_near_miss: $n=465$, Target--Full accuracy -0.003 [-0.022,+0.014].
