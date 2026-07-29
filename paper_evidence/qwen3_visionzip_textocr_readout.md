# Qwen3 VisionZip TextOCR-Hard Native-Port Readout

This report summarizes the matched-budget Qwen3 VisionZip native-port run. The VisionZip row uses the same TextOCR-Hard 1000-probe input and the same fixed 802816-pixel Qwen image policy as the main Qwen rows.

## Verdict

The native Qwen3 VisionZip port is runnable and matched-budget on TextOCR-Hard, but its merge-aware provenance needs two readings. LineageECR propagates every source cell into its contextual merge and is therefore 1.0; AnchorECR measures only the representative output locations. VisionZip lowers hFPR relative to Target 0.30 but does not match its accuracy. This is a fair external-baseline result, not a state-of-the-art claim.

## Matched-Budget Rows

| method | comparison_type | n | accuracy | hFPR | mean_actual_keep_ratio | ECR | AnchorECR | center_recall | patch_recall | matched_budget_to_target |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen Target 0.30 (ours) | ours | 1000.000000 | 0.798000 | 0.222000 | 0.300558 | 0.604305 | 0.604305 | 0.668000 | 0.499500 | True |
| Qwen VisionZip 0.30 (native port) | official_algorithm_port | 1000.000000 | 0.764000 | 0.184000 | 0.300558 | 1.000000 | 0.868882 | 1.000000 | 1.000000 | True |
| Qwen Random 0.30 | sanity_baseline | 1000.000000 | 0.733000 | 0.184000 | 0.300558 | 0.308637 | 0.308637 | 0.297000 | 0.308250 | True |

## Deltas vs Ours

| method | accuracy_delta_vs_ours | hfpr_delta_vs_ours | ecr_delta_vs_ours |
| --- | --- | --- | --- |
| Qwen Target 0.30 (ours) | 0.000000 | 0.000000 | 0.000000 |
| Qwen VisionZip 0.30 (native port) | -0.034000 | -0.038000 | 0.395695 |
| Qwen Random 0.30 | -0.065000 | -0.038000 | -0.295668 |

## Excluded Run

The following completed run is excluded from matched-budget claims because its image pixel policy differs from the main Qwen 802816 setting.

| method | mean_full_visual_tokens | mean_kept_visual_tokens | accuracy | ECR | exclusion_reason |
| --- | --- | --- | --- | --- | --- |
| Qwen VisionZip 0.30 (unmatched pixel policy) | 714.222000 | 214.778000 | 0.738000 | 1.000000 | uses min=50176,max=802816 and therefore has fewer visual tokens than the main Qwen 802816 runs |
