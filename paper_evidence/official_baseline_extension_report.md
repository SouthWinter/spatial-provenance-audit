# Official Baseline Extension Check

Scope: TextOCR-HardYesNo. This report separates official-algorithm ports from protocol-compatible proxies.

## Status Matrix

| model | method | implementation | status | evidence |
|---|---|---|---|---|
| LLaVA-1.5-7B | VisionZip | official-algorithm port | done | 0.20/0.30/0.40/0.50 budget curve exists |
| LLaVA-1.5-7B | FastV | official-algorithm port | done | 0.20/0.30/0.40/0.50 budget curve exists |
| InternVL3.5-8B | VisionZip | not claimed | unsupported | official CLIP CLS-attention path is not available in current InternVL backend |
| InternVL3.5-8B | FastV | not claimed | not implemented | no InternVL decoder-layer official FastV branch exists |
| InternVL3.5-8B | FastV/TopV-style proxy | protocol proxy | done | calibrated `embed0p50` row exists; must not be called official |

## LLaVA Official Budget Curve

| method | ratio | acc | hFPR | pos acc | neg acc | yes rate | keep | Lineage ECR | Anchor ECR | path |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ours_protected_embed | 0.200 | 0.623 | 0.542 | 0.788 | 0.458 | 0.665 | 0.201 | 1.000 | 1.000 | `runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_embed_protected_topk_0p20` |
| VisionZip | 0.200 | 0.554 | 0.780 | 0.888 | 0.220 | 0.834 | 0.201 | 1.000 | 0.336 | `runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_visionzip_0p20` |
| FastV | 0.200 | 0.500 | 0.000 | 0.000 | 1.000 | 0.000 | 0.201 | 0.270 | 0.270 | `runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_fastv_0p20` |
| ours_protected_embed | 0.300 | 0.627 | 0.440 | 0.694 | 0.560 | 0.567 | 0.300 | 1.000 | 1.000 | `runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_embed_protected_topk_0p30` |
| VisionZip | 0.300 | 0.555 | 0.734 | 0.844 | 0.266 | 0.789 | 0.300 | 1.000 | 0.466 | `runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_visionzip_0p30` |
| FastV | 0.300 | 0.500 | 0.000 | 0.000 | 1.000 | 0.000 | 0.300 | 0.376 | 0.376 | `runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_fastv_0p30` |
| ours_protected_embed | 0.400 | 0.661 | 0.298 | 0.620 | 0.702 | 0.459 | 0.401 | 1.000 | 1.000 | `runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_embed_protected_topk_0p40` |
| VisionZip | 0.400 | 0.580 | 0.650 | 0.810 | 0.350 | 0.730 | 0.401 | 1.000 | 0.575 | `runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_visionzip_0p40` |
| FastV | 0.400 | 0.500 | 0.000 | 0.000 | 1.000 | 0.000 | 0.401 | 0.494 | 0.494 | `runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_fastv_0p40` |
| ours_protected_embed | 0.500 | 0.667 | 0.308 | 0.642 | 0.692 | 0.475 | 0.500 | 1.000 | 1.000 | `runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_embed_protected_topk_0p50` |
| VisionZip | 0.500 | 0.542 | 0.834 | 0.918 | 0.166 | 0.876 | 0.500 | 1.000 | 0.661 | `runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_visionzip_0p50` |
| FastV | 0.500 | 0.500 | 0.000 | 0.000 | 1.000 | 0.000 | 0.500 | 0.604 | 0.604 | `runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_fastv_0p50` |

## LLaVA Paired Checks vs Ours

| right method | ratio | n | acc diff | acc p | hFPR diff | hFPR p | reading |
|---|---:|---:|---:|---:|---:|---:|---|
| FastV | 0.200 | 1000 | +0.123 | <1e-4 | +0.542 | <1e-4 | acc higher; hFPR higher |
| VisionZip | 0.200 | 1000 | +0.069 | <1e-4 | -0.238 | <1e-4 | acc higher; hFPR lower |
| FastV | 0.300 | 1000 | +0.127 | <1e-4 | +0.440 | <1e-4 | acc higher; hFPR higher |
| VisionZip | 0.300 | 1000 | +0.072 | <1e-4 | -0.294 | <1e-4 | acc higher; hFPR lower |
| FastV | 0.400 | 1000 | +0.161 | <1e-4 | +0.298 | <1e-4 | acc higher; hFPR higher |
| VisionZip | 0.400 | 1000 | +0.081 | <1e-4 | -0.352 | <1e-4 | acc higher; hFPR lower |
| FastV | 0.500 | 1000 | +0.167 | <1e-4 | +0.308 | <1e-4 | acc higher; hFPR higher |
| VisionZip | 0.500 | 1000 | +0.125 | <1e-4 | -0.526 | <1e-4 | acc higher; hFPR lower |

## InternVL Official-Port Check

| method | implementation | status | acc | hFPR | keep | ECR | path / note |
|---|---|---|---:|---:|---:|---:|---|
| FastV/TopV-style proxy | protocol proxy | done | 0.618 | 0.246 | 0.500 | 0.792 | `runs/internvl_textocr_hard/calibrated_test_embed0p50_devthr` |
| ours_soft_evidence | ours | done | 0.647 | 0.175 | 0.500 | 0.893 | `runs/internvl_textocr_hard/calibrated_test_target_soft_evidence0p50_b0p05_hfprconstr_devthr` |
| VisionZip | not claimed | unsupported |  |  |  |  | Official VisionZip is CLIP-ViT CLS-attention plus contextual merge. Current InternVL-HF backend exposes tiled image features/placeholder tokens, not the same official CLS-attention path. |
| FastV | not claimed | not implemented |  |  |  |  | Official FastV is a LLaVA decoder-layer attention hook. InternVL backend has no method-specific FastV branch; existing InternVL rows remain proxies unless this is separately ported and validated. |

## Reviewer-Facing Readout

- LLaVA now has the official-baseline extension check: both VisionZip and FastV are evaluated at 0.20/0.30/0.40/0.50 under the same TextOCR-Hard protocol.
- InternVL should remain labelled as proxy-only for external methods. Calling its `embed0p50` row official would be misleading.
- FastV hFPR must be interpreted with positive accuracy and yes rate: on LLaVA it collapses to all-no, so hFPR=0 is not a useful standalone win.
- For the paper, use LLaVA official ports as the main external-method evidence and InternVL proxy rows as cross-model sanity checks.

## Source Hooks

- LLaVA official paths: `recap/llava_pruned_backend.py` has method-specific `visionzip` and `fastv` branches.
- InternVL backend: `recap/internvl_pruned_backend.py` only uses generic selector dispatch and has no method-specific official branch.
- CLI wording: `run-llava-pruned` advertises LLaVA-only official ports, while `run-internvl-pruned` lists only generic non-oracle selector variants.
