# Full Development Hard-Negative Human-QC and Sensitivity Audit

All 500 development-set hard negatives received final human QC. The original 100 rows and the adjudicated 400-row extension are disjoint.

## Label and Source-Region Quality

- Strictly valid near-miss probes: 449/500.
- Target absence confirmed after normalization: 486/500.
- Target observed in the same image: 0/500.
- Source not visually identifiable: 37/500.
- Unclear: 14/500.

The target-absence scope audits negative-label validity. The strict scope additionally requires a visible source and matching source box, which is necessary for interpreting NegSRC.

## Independent Reliability

- A second non-author annotator independently reviewed a frozen 100-row subset before adjudication.
- QC-decision agreement/kappa: 0.820/0.323.
- Valid-vs-non-valid agreement/kappa: 0.860/0.453.
- Exact five-field agreement: 0.780; 22 rows were adjudicated.

## hFPR Sensitivity

| Model | Method | All $n$/hFPR | Absence-confirmed $n$/hFPR | Strict $n$/hFPR | Max $|\Delta|$ |
|---|---|---:|---:|---:|---:|
| Qwen3-VL-8B | Full | 500/0.236 | 486/0.241 | 449/0.249 | 0.013 |
| Qwen3-VL-8B | Target (30%) | 500/0.222 | 486/0.222 | 449/0.227 | 0.005 |
| Qwen3-VL-8B | Random (30%) | 500/0.184 | 486/0.181 | 449/0.187 | 0.003 |
| Qwen3-VL-8B | Grid (30%) | 500/0.164 | 486/0.167 | 449/0.176 | 0.012 |
| Qwen3-VL-8B | VisionZip (30%) | 500/0.184 | 486/0.185 | 449/0.192 | 0.008 |
| LLaVA-1.5-7B | Full | 500/0.304 | 486/0.296 | 449/0.292 | 0.012 |
| LLaVA-1.5-7B | Protected (40%) | 500/0.298 | 486/0.292 | 449/0.290 | 0.008 |
| LLaVA-1.5-7B | Random (40%) | 500/0.256 | 486/0.251 | 449/0.245 | 0.011 |
| LLaVA-1.5-7B | Target (40%) | 500/0.316 | 486/0.311 | 449/0.312 | 0.005 |
| LLaVA-1.5-7B | SCOPE (40%) | 500/0.508 | 486/0.502 | 449/0.501 | 0.007 |
| LLaVA-1.5-7B | AnchorPrune (40%) | 500/0.570 | 486/0.566 | 449/0.566 | 0.004 |
| LLaVA-1.5-7B | CoIn (40%) | 500/0.632 | 486/0.630 | 449/0.624 | 0.008 |
| LLaVA-1.5-7B | VisionZip (40%) | 500/0.650 | 486/0.648 | 449/0.653 | 0.003 |
| InternVL3.5-8B | Full | 500/0.282 | 486/0.286 | 449/0.283 | 0.004 |
| InternVL3.5-8B | Soft evidence (50%) | 500/0.520 | 486/0.523 | 449/0.528 | 0.008 |
| InternVL3.5-8B | Random (50%) | 500/0.482 | 486/0.488 | 449/0.492 | 0.010 |
| InternVL3.5-8B | Grid (50%) | 500/0.462 | 486/0.469 | 449/0.468 | 0.007 |

The all-development rows remain the original readout. The two filtered columns are post-QC sensitivity analyses; they do not replace or validate the locked confirmation set.
