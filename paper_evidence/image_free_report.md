# TextOCR-Hard Image-Free Control Audit

Blank images preserve each source image's dimensions and aspect ratio. Each mismatch seed is a collision-free image-group derangement; paired positive and negative probes share the same wrong image. Thresholds are frozen from the matched-image protocol.

| Model | Condition | n | Acc. | Pos. acc. | hFPR | AUROC (95% CI) | Pair margin win | $\Delta$ Acc. vs. matched (95% CI) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-VL-8B | matched_image_full | 1000 | 0.783 | 0.814 | 0.248 | 0.856 [0.834,0.878] | 0.924 | -- |
| Qwen3-VL-8B | blank | 1000 | 0.502 | 0.006 | 0.002 | 0.574 [0.548,0.601] | 0.608 | -0.281 [-0.304,-0.258] |
| Qwen3-VL-8B | image_mismatch_seed101 | 1000 | 0.504 | 0.014 | 0.006 | 0.414 [0.392,0.434] | 0.346 | -0.279 [-0.302,-0.256] |
| Qwen3-VL-8B | image_mismatch_seed202 | 1000 | 0.502 | 0.022 | 0.018 | 0.386 [0.365,0.406] | 0.284 | -0.281 [-0.305,-0.257] |
| Qwen3-VL-8B | image_mismatch_seed303 | 1000 | 0.504 | 0.020 | 0.012 | 0.398 [0.378,0.418] | 0.324 | -0.279 [-0.302,-0.256] |
| LLaVA-1.5-7B | matched_image_full | 1000 | 0.625 | 0.566 | 0.316 | 0.666 [0.646,0.686] | 0.768 | -- |
| LLaVA-1.5-7B | blank | 1000 | 0.503 | 0.008 | 0.002 | 0.693 [0.668,0.721] | 0.746 | -0.122 [-0.143,-0.101] |
| LLaVA-1.5-7B | image_mismatch_seed101 | 1000 | 0.556 | 0.342 | 0.230 | 0.567 [0.546,0.587] | 0.616 | -0.069 [-0.097,-0.042] |
| LLaVA-1.5-7B | image_mismatch_seed202 | 1000 | 0.555 | 0.338 | 0.228 | 0.562 [0.542,0.582] | 0.614 | -0.070 [-0.096,-0.045] |
| LLaVA-1.5-7B | image_mismatch_seed303 | 1000 | 0.558 | 0.360 | 0.244 | 0.575 [0.556,0.595] | 0.626 | -0.067 [-0.093,-0.041] |
| InternVL3.5-8B | matched_image_full | 536 | 0.646 | 0.597 | 0.306 | 0.687 [0.653,0.722] | 0.731 | -- |
| InternVL3.5-8B | blank | 536 | 0.618 | 0.817 | 0.582 | 0.676 [0.643,0.709] | 0.701 | -0.028 [-0.073,+0.017] |
| InternVL3.5-8B | image_mismatch_seed101 | 536 | 0.532 | 0.146 | 0.082 | 0.530 [0.501,0.560] | 0.549 | -0.114 [-0.155,-0.071] |
| InternVL3.5-8B | image_mismatch_seed202 | 536 | 0.515 | 0.146 | 0.116 | 0.515 [0.489,0.541] | 0.519 | -0.131 [-0.172,-0.090] |
| InternVL3.5-8B | image_mismatch_seed303 | 536 | 0.528 | 0.164 | 0.108 | 0.528 [0.501,0.556] | 0.541 | -0.118 [-0.157,-0.076] |

## Mismatch Seed Stability

| Model | Seeds | Acc. mean [range] | hFPR mean [range] | AUROC mean [range] | $\Delta$ Acc. mean [range] |
|---|---:|---:|---:|---:|---:|
| Qwen3-VL-8B | 3 | 0.503 [0.502,0.504] | 0.012 [0.006,0.018] | 0.399 [0.386,0.414] | -0.280 [-0.281,-0.279] |
| LLaVA-1.5-7B | 3 | 0.556 [0.555,0.558] | 0.234 [0.228,0.244] | 0.568 [0.562,0.575] | -0.069 [-0.070,-0.067] |
| InternVL3.5-8B | 3 | 0.525 [0.515,0.532] | 0.102 [0.082,0.116] | 0.524 [0.515,0.530] | -0.121 [-0.131,-0.114] |

## Lexically Plausible Decoy Subset

This subset retains image pairs whose decoy occurs as a real OCR token in another TextOCR image, reducing the real-string versus synthetic-string confound.

| Model | Condition | n | Acc. | Pos. acc. | hFPR | AUROC (95% CI) | $\Delta$ Acc. vs. matched (95% CI) |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3-VL-8B | matched_image_full | 128 | 0.672 | 0.766 | 0.422 | 0.813 [0.746,0.879] | -- |
| Qwen3-VL-8B | blank | 128 | 0.500 | 0.000 | 0.000 | 0.505 [0.427,0.584] | -0.172 [-0.242,-0.102] |
| Qwen3-VL-8B | image_mismatch_seed101 | 128 | 0.492 | 0.000 | 0.016 | 0.381 [0.314,0.440] | -0.180 [-0.250,-0.109] |
| Qwen3-VL-8B | image_mismatch_seed202 | 128 | 0.516 | 0.047 | 0.016 | 0.388 [0.317,0.451] | -0.156 [-0.227,-0.086] |
| Qwen3-VL-8B | image_mismatch_seed303 | 128 | 0.484 | 0.031 | 0.062 | 0.405 [0.344,0.463] | -0.188 [-0.258,-0.117] |
| LLaVA-1.5-7B | matched_image_full | 128 | 0.609 | 0.594 | 0.375 | 0.637 [0.584,0.698] | -- |
| LLaVA-1.5-7B | blank | 128 | 0.508 | 0.016 | 0.000 | 0.595 [0.517,0.677] | -0.102 [-0.156,-0.047] |
| LLaVA-1.5-7B | image_mismatch_seed101 | 128 | 0.539 | 0.344 | 0.266 | 0.574 [0.526,0.629] | -0.070 [-0.141,+0.000] |
| LLaVA-1.5-7B | image_mismatch_seed202 | 128 | 0.516 | 0.375 | 0.344 | 0.537 [0.484,0.591] | -0.094 [-0.164,-0.023] |
| LLaVA-1.5-7B | image_mismatch_seed303 | 128 | 0.578 | 0.391 | 0.234 | 0.573 [0.516,0.637] | -0.031 [-0.102,+0.039] |
| InternVL3.5-8B | matched_image_full | 66 | 0.712 | 0.667 | 0.242 | 0.743 [0.650,0.837] | -- |
| InternVL3.5-8B | blank | 66 | 0.500 | 0.848 | 0.848 | 0.572 [0.466,0.682] | -0.212 [-0.318,-0.106] |
| InternVL3.5-8B | image_mismatch_seed101 | 66 | 0.470 | 0.091 | 0.152 | 0.475 [0.385,0.566] | -0.242 [-0.348,-0.136] |
| InternVL3.5-8B | image_mismatch_seed202 | 66 | 0.439 | 0.121 | 0.242 | 0.463 [0.377,0.546] | -0.273 [-0.409,-0.136] |
| InternVL3.5-8B | image_mismatch_seed303 | 66 | 0.530 | 0.303 | 0.242 | 0.540 [0.456,0.624] | -0.182 [-0.288,-0.091] |

## Blank-Image Lexical Strata

| Model | Stratum | Bucket | Pairs | Mean positive-negative margin | Positive margin win |
|---|---|---|---:|---:|---:|
| Qwen3-VL-8B | decoy_seen_elsewhere_in_TextOCR | seen | 64 | 0.040 | 0.594 |
| Qwen3-VL-8B | decoy_seen_elsewhere_in_TextOCR | unseen | 436 | 0.196 | 0.610 |
| Qwen3-VL-8B | edit_type | other | 6 | 0.646 | 0.833 |
| Qwen3-VL-8B | edit_type | single_deletion_or_insertion | 118 | 0.269 | 0.653 |
| Qwen3-VL-8B | edit_type | single_substitution | 376 | 0.140 | 0.590 |
| Qwen3-VL-8B | source_length | 4_or_less | 230 | 0.131 | 0.587 |
| Qwen3-VL-8B | source_length | 5_to_7 | 231 | 0.249 | 0.623 |
| Qwen3-VL-8B | source_length | 8_or_more | 39 | 0.011 | 0.641 |
| Qwen3-VL-8B | source_shape | alpha | 415 | 0.221 | 0.631 |
| Qwen3-VL-8B | source_shape | alpha+digit | 11 | -0.053 | 0.455 |
| Qwen3-VL-8B | source_shape | alpha+digit+punct | 4 | 0.031 | 0.500 |
| Qwen3-VL-8B | source_shape | alpha+punct | 26 | -0.103 | 0.462 |
| Qwen3-VL-8B | source_shape | digit | 31 | -0.001 | 0.516 |
| Qwen3-VL-8B | source_shape | digit+punct | 13 | -0.016 | 0.538 |
| LLaVA-1.5-7B | decoy_seen_elsewhere_in_TextOCR | seen | 64 | 0.051 | 0.688 |
| LLaVA-1.5-7B | decoy_seen_elsewhere_in_TextOCR | unseen | 436 | 0.102 | 0.755 |
| LLaVA-1.5-7B | edit_type | other | 6 | 0.177 | 0.833 |
| LLaVA-1.5-7B | edit_type | single_deletion_or_insertion | 118 | 0.066 | 0.661 |
| LLaVA-1.5-7B | edit_type | single_substitution | 376 | 0.103 | 0.771 |
| LLaVA-1.5-7B | source_length | 4_or_less | 230 | 0.096 | 0.735 |
| LLaVA-1.5-7B | source_length | 5_to_7 | 231 | 0.103 | 0.775 |
| LLaVA-1.5-7B | source_length | 8_or_more | 39 | 0.046 | 0.641 |
| LLaVA-1.5-7B | source_shape | alpha | 415 | 0.108 | 0.764 |
| LLaVA-1.5-7B | source_shape | alpha+digit | 11 | 0.027 | 0.636 |
| LLaVA-1.5-7B | source_shape | alpha+digit+punct | 4 | 0.025 | 0.750 |
| LLaVA-1.5-7B | source_shape | alpha+punct | 26 | 0.057 | 0.654 |
| LLaVA-1.5-7B | source_shape | digit | 31 | 0.020 | 0.677 |
| LLaVA-1.5-7B | source_shape | digit+punct | 13 | 0.016 | 0.615 |
| InternVL3.5-8B | decoy_seen_elsewhere_in_TextOCR | seen | 33 | 0.056 | 0.576 |
| InternVL3.5-8B | decoy_seen_elsewhere_in_TextOCR | unseen | 235 | 0.210 | 0.719 |
| InternVL3.5-8B | edit_type | other | 4 | 0.380 | 0.750 |
| InternVL3.5-8B | edit_type | single_deletion_or_insertion | 68 | 0.101 | 0.574 |
| InternVL3.5-8B | edit_type | single_substitution | 196 | 0.219 | 0.745 |
| InternVL3.5-8B | source_length | 4_or_less | 119 | 0.128 | 0.647 |
| InternVL3.5-8B | source_length | 5_to_7 | 121 | 0.233 | 0.752 |
| InternVL3.5-8B | source_length | 8_or_more | 28 | 0.280 | 0.714 |
| InternVL3.5-8B | source_shape | alpha | 213 | 0.226 | 0.732 |
| InternVL3.5-8B | source_shape | alpha+digit | 9 | 0.052 | 0.444 |
| InternVL3.5-8B | source_shape | alpha+digit+punct | 10 | 0.053 | 0.600 |
| InternVL3.5-8B | source_shape | alpha+punct | 11 | 0.103 | 0.636 |
| InternVL3.5-8B | source_shape | digit | 17 | -0.024 | 0.529 |
| InternVL3.5-8B | source_shape | digit+punct | 8 | 0.163 | 0.750 |
