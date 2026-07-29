# Repeated Exclusive LLaVA Selector Timing

Three fresh-process repetitions use the same first 100 probes from the image-disjoint TextOCR-Hard confirmation split, one A800 GPU, float16, eager attention, and rotated method order. The first 5 probes of each process are discarded as warm-up.

## Summary

| method | keep | vision ms | selector ms | LLM ms | total ms | speedup |
|---|---:|---:|---:|---:|---:|---:|
| full | 1.000 | 9.9 +/- 0.1 | 0.8 +/- 0.0 | 63.6 +/- 0.1 | 75.3 +/- 0.1 | 1.00x |
| protected | 0.400 | 10.4 +/- 0.2 | 6.1 +/- 0.1 | 38.0 +/- 0.3 | 55.3 +/- 0.5 | 1.36x |
| target | 0.400 | 10.1 +/- 0.1 | 5.3 +/- 0.1 | 38.0 +/- 0.1 | 54.4 +/- 0.2 | 1.39x |
| scope | 0.400 | 9.7 +/- 0.2 | 42.2 +/- 1.0 | 37.6 +/- 0.2 | 90.7 +/- 1.3 | 0.83x |
| coin | 0.400 | 10.0 +/- 0.2 | 84.6 +/- 1.8 | 37.5 +/- 0.1 | 133.0 +/- 2.0 | 0.57x |

Values are mean +/- sample standard deviation across three repetitions; each repetition averages 95 post-warm-up probes.

## Per-Run Means

| method | rep | total ms | selector ms | LLM ms |
|---|---:|---:|---:|---:|
| full | 1 | 75.33 | 0.85 | 63.50 |
| full | 2 | 75.43 | 0.85 | 63.59 |
| full | 3 | 75.14 | 0.83 | 63.61 |
| protected | 1 | 55.81 | 6.19 | 38.20 |
| protected | 2 | 55.37 | 6.13 | 38.18 |
| protected | 3 | 54.80 | 5.95 | 37.73 |
| target | 1 | 54.46 | 5.30 | 38.10 |
| target | 2 | 54.15 | 5.30 | 37.95 |
| target | 3 | 54.46 | 5.39 | 38.06 |
| scope | 1 | 92.19 | 43.31 | 37.64 |
| scope | 2 | 89.88 | 41.77 | 37.33 |
| scope | 3 | 90.12 | 41.56 | 37.69 |
| coin | 1 | 131.40 | 83.29 | 37.40 |
| coin | 2 | 135.24 | 86.66 | 37.48 |
| coin | 3 | 132.23 | 83.87 | 37.67 |

## Readout

- Protected and Target reduce LLM-prefix time; their total benefit depends on selector cost.
- SCOPE and CoIn measure their paper-defined coverage selection inside the online path.
- These are single-sample prefill/likelihood timings, not end-to-end long-generation speedups.
