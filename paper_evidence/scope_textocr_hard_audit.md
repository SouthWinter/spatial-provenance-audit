# SCOPE TextOCR-Hard Audit

## Implementation Fidelity

- Upstream: `kinredon/SCOPE@6bf73069e0d61307051cfda8e25925bc7b7afdd9`.
- Official default: `alpha=1`, multiplicative saliency--coverage combination.
- Exact-index parity audit: **pass**.
- Extracted official source SHA-256: `67e0df9bc11d75290f471b7e3bf0f033d8e9028631ab453e9d26afda7cfb6971`.

## Development Quality

| method | probes | keep | accuracy | hFPR | ECR | CenterR | PatchR |
|---|---:|---:|---:|---:|---:|---:|---:|
| SCOPE (40%) | 1000 | 0.401 | 0.598 | 0.508 | 0.642 | 0.660 | 0.625 |

## Locked Image-Disjoint Confirmation

| method | probes | keep | accuracy | hFPR | ECR |
|---|---:|---:|---:|---:|---:|
| Full prefix | 1000 | 1.000 | 0.625 | 0.316 | 1.000 |
| Target (40%) | 1000 | 0.401 | 0.624 | 0.574 | 0.457 |
| Protected (40%) | 1000 | 0.401 | 0.643 | 0.352 | 1.000 |
| Random (40%) | 1000 | 0.401 | 0.669 | 0.256 | 0.381 |
| SCOPE (40%) | 1000 | 0.401 | 0.586 | 0.562 | 0.614 |
| VisionZip (40%) | 1000 | 0.401 | 0.582 | 0.670 | 0.563 |

## Paired Development Comparisons

| comparison | delta accuracy [95% CI] | p | delta hFPR [95% CI] | p |
|---|---:|---:|---:|---:|
| llava_protected0p40_vs_scope_official | +0.063 [+0.031, +0.094] | 0.0001307 | -0.210 [-0.254, -0.164] | 8.287e-18 |
| llava_target0p40_vs_scope_official | +0.025 [-0.005, +0.055] | 0.1219 | -0.012 [-0.060, +0.036] | 0.6832 |
| llava_scope0p40_vs_visionzip_official | +0.018 [-0.010, +0.045] | 0.2292 | -0.142 [-0.186, -0.100] | 1.178e-10 |

## Paired Confirmation Comparisons

| comparison | delta accuracy [95% CI] | p | delta hFPR [95% CI] | p |
|---|---:|---:|---:|---:|
| scope_vs_full | -0.039 [-0.071, -0.007] | 0.0176 | +0.246 [+0.202, +0.290] | 4.35e-26 |
| scope_vs_target | -0.038 [-0.068, -0.008] | 0.01452 | -0.012 [-0.057, +0.033] | 0.6683 |
| protected_vs_scope | +0.057 [+0.025, +0.089] | 0.0006425 | -0.210 [-0.256, -0.164] | 4.157e-17 |
| random_vs_scope | +0.083 [+0.050, +0.117] | 1.201e-06 | -0.306 [-0.351, -0.261] | 5.631e-35 |
| scope_vs_visionzip | +0.004 [-0.023, +0.031] | 0.826 | -0.108 [-0.150, -0.068] | 3.291e-07 |

## Repeated Exclusive Single-Sample Timing

Each method was measured in three fresh processes on the same 100 confirmation probes. The first five probes per process were discarded, leaving 95 timed probes per repetition. Method order was rotated across repetitions; all rows use one A800 GPU, eager attention, float16, and the same pruning backend.

| method | keep | vision ms | selector/materialize ms | LLM ms | total ms | speedup |
|---|---:|---:|---:|---:|---:|---:|
| full | 1.000 | 10.2 +/- 0.2 | 0.9 +/- 0.0 | 63.6 +/- 0.1 | 75.8 +/- 0.4 | 1.00x |
| protected | 0.400 | 10.0 +/- 0.1 | 5.9 +/- 0.0 | 37.8 +/- 0.1 | 54.5 +/- 0.1 | 1.39x |
| scope | 0.400 | 9.6 +/- 0.5 | 41.8 +/- 3.1 | 37.9 +/- 0.5 | 90.5 +/- 4.1 | 0.84x |

## Claim Boundary

- Protected uses evidence boxes; its comparison with box-free SCOPE is not resource-symmetric.
- On development data, box-free Target and SCOPE are tied on answer metrics; on the locked confirmation set, Target is 3.8 accuracy points higher while hFPR remains tied.
- SCOPE retains more annotated evidence than box-free Target on both splits, but evidence coverage alone does not guarantee lower answer risk.
- Across three fresh-process timing repetitions, SCOPE lowers LLM prefill time but its greedy coverage selector makes total single-sample forward slower than the full-prefix reference.
