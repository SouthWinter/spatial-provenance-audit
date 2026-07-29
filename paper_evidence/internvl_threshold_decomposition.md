# InternVL Threshold Decomposition

All thresholds are selected on the fixed 464-probe development split and evaluated on the same 536-probe test split.
The Full-only shared threshold is 2.043; the pooled shared threshold is 2.323.

| Method | Test AUROC | Per-method Acc. | Per-method hFPR | Full-shared Acc. | Full-shared hFPR | Pooled Acc. | Pooled hFPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full | 0.687 | 0.646 | 0.306 | 0.646 | 0.306 | 0.638 | 0.198 |
| Target | 0.695 | 0.640 | 0.179 | 0.618 | 0.522 | 0.655 | 0.340 |
| Grid | 0.699 | 0.647 | 0.254 | 0.623 | 0.474 | 0.644 | 0.287 |
| Random | 0.677 | 0.614 | 0.336 | 0.604 | 0.485 | 0.614 | 0.336 |
| Soft evidence | 0.696 | 0.653 | 0.328 | 0.616 | 0.530 | 0.659 | 0.340 |

Per-method thresholds conflate selector and operating-point calibration. Full-shared and pooled-shared columns isolate behavior under one common threshold, while AUROC is threshold-free.
