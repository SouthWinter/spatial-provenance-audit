# LLaVA-1.5-7B TextOCR-Hard pruning curve

Dataset: `data/textocr_val_hard_probes_500img.jsonl`  
Model: `llava-hf/llava-1.5-7b-hf`  
Selector: `target_embed_topk`

| run | acc | hFPR | keep | ECR | CenterR | PatchR | mean forward ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| full visual | 0.626 | 0.304 | 1.000 | - | - | - | - |
| target@0.30 | 0.630 | 0.420 | 0.300 | 0.336 | 0.346 | 0.333 | 39.7 |
| target@0.40 | 0.662 | 0.316 | 0.401 | 0.397 | 0.404 | 0.394 | 45.0 |
| target@0.50 | 0.665 | 0.304 | 0.500 | 0.472 | 0.482 | 0.469 | 46.0 |

Paired bootstrap / McNemar summary:

| comparison | acc diff | acc 95% CI | acc p | hFPR diff | hFPR 95% CI | hFPR p |
|---|---:|---|---:|---:|---|---:|
| target@0.30 vs full | +0.004 | [-0.029, +0.039] | 0.8634 | +0.116 | [+0.067, +0.164] | 9.02e-06 |
| target@0.40 vs full | +0.036 | [+0.006, +0.070] | 0.0273 | +0.012 | [-0.035, +0.056] | 0.666 |
| target@0.50 vs full | +0.039 | [+0.010, +0.067] | 0.0094 | +0.000 | [-0.043, +0.041] | 1.0 |
| target@0.40 vs target@0.30 | +0.032 | [+0.006, +0.058] | 0.0178 | -0.104 | [-0.142, -0.065] | 1.317e-07 |
| target@0.50 vs target@0.40 | +0.003 | [-0.018, +0.023] | 0.8482 | -0.012 | [-0.041, +0.017] | 0.4966 |

Positive/negative split:

| run | pos acc | neg acc | neg yes / hFPR | overall yes rate |
|---|---:|---:|---:|---:|
| full visual | 0.556 | 0.696 | 0.304 | 0.430 |
| target@0.30 | 0.680 | 0.580 | 0.420 | 0.550 |
| target@0.40 | 0.640 | 0.684 | 0.316 | 0.478 |
| target@0.50 | 0.634 | 0.696 | 0.304 | 0.469 |

Takeaway:

`target@0.30` is too aggressive for LLaVA: it raises positive recall but causes a significant hard-negative hallucination increase. `target@0.40` is the stronger efficiency point, with significant accuracy gain and non-significant hFPR increase. `target@0.50` is the safest LLaVA main point: significant accuracy gain with hFPR exactly matching the full-visual baseline.

