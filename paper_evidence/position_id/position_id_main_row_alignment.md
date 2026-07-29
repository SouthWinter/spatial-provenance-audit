# Position-ID audit to main-table alignment

## Resolution

The earlier position-policy audit was not numerically comparable to the main rows because it recomputed selector masks and the LLaVA rerun used a different dtype. The corrected audit fixes all non-position variables:

- Compact scores and traces are the original main-table artifacts.
- Preserve replays the exact per-sample main-row kept indices.
- Both replay runs use bfloat16, matching the main runs.
- Probe order, sample IDs, masks, and evidence coverage are checked pairwise.
- Compact uses the model's ordinary shortened-prefix path; Preserve changes only logical position IDs.

## Provenance

| Model and row | Compact score source | Main-mask file | Preserve score source |
|---|---|---|---|
| LLaVA-1.5-7B Protected (40%) | `runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_embed_protected_topk_0p40/probe_scores.jsonl` | `runs/position_id_audit/main_mask_replay/indices/llava_protected40_main_masks.jsonl` | `runs/position_id_audit/main_mask_replay/llava_protected40_preserve_bf16/probe_scores.jsonl` |
| InternVL3.5-8B Soft evidence (50%) | `runs/internvl_textocr_hard/internvl35_8b_textocr_hard_full1000_target_embed_soft_evidence_topk_0p50_b0p05/probe_scores.jsonl` | `runs/position_id_audit/main_mask_replay/indices/internvl_soft50_main_masks.jsonl` | `runs/position_id_audit/main_mask_replay/internvl_soft50_preserve_bf16/probe_scores.jsonl` |

Mask-file SHA256 values:

- LLaVA: `fa9b6d3ff07276e27362fa16955c72dc413159124296b3a7c19356b0dfdc3156`
- InternVL: `c77d7c21c5e9fd8c83d24e8b9351d079996bf911590ed8414dad6f45e55eee70`

## Aligned results

| Model | Policy | Evaluation | Acc. | hFPR | AUROC | Shared-threshold flips |
|---|---|---|---:|---:|---:|---:|
| LLaVA Protected (40%) | Compact | all 1000, raw | 0.661 | 0.298 | -- | 279/1000 raw |
| LLaVA Protected (40%) | Preserve | all 1000, raw | 0.654 | 0.362 | -- | 279/1000 raw |
| InternVL Soft evidence (50%) | Compact | held-out 536, own-dev | 0.653 | 0.328 | 0.696 | 101/536 |
| InternVL Soft evidence (50%) | Preserve | held-out 536, own-dev | 0.647 | 0.343 | 0.678 | 101/536 |

For InternVL, applying the Compact-derived threshold to Preserve gives 0.647 accuracy and 0.272 hFPR. For every pair, kept-index mismatches, ECR mismatches, and sample-ID mismatches are all zero.

The older fresh-selector reruns under `runs/position_id_audit/llava15_7b_protected40_*` and `runs/position_id_audit/internvl35_8b_soft50_*` are retained only as historical diagnostics and must not be used for manuscript numbers.
