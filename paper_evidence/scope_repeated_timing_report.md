# Repeated Exclusive LLaVA Timing

Three fresh-process repetitions use the same first 100 image-disjoint TextOCR-Hard confirmation probes, one A800 GPU, float16, eager attention, and rotated method order. The first 5 probes of each process are discarded as warm-up.

## Summary

| method | keep | vision ms | selector ms | LLM ms | total ms | speedup |
|---|---:|---:|---:|---:|---:|---:|
| full | 1.000 | 10.2 +/- 0.2 | 0.9 +/- 0.0 | 63.6 +/- 0.1 | 75.8 +/- 0.4 | 1.00x |
| protected | 0.400 | 10.0 +/- 0.1 | 5.9 +/- 0.0 | 37.8 +/- 0.1 | 54.5 +/- 0.1 | 1.39x |
| scope | 0.400 | 9.6 +/- 0.5 | 41.8 +/- 3.1 | 37.9 +/- 0.5 | 90.5 +/- 4.1 | 0.84x |

Values are mean +/- sample standard deviation across three repetitions; each repetition averages 95 post-warm-up probes.

## Per-Run Means

| method | rep | total ms | selector ms | LLM ms |
|---|---:|---:|---:|---:|
| full | 1 | 75.93 | 0.88 | 63.67 |
| full | 2 | 75.38 | 0.86 | 63.48 |
| full | 3 | 76.09 | 0.88 | 63.72 |
| protected | 1 | 54.67 | 5.90 | 37.86 |
| protected | 2 | 54.55 | 5.92 | 37.73 |
| protected | 3 | 54.42 | 5.91 | 37.85 |
| scope | 1 | 88.61 | 40.22 | 37.85 |
| scope | 2 | 87.70 | 39.86 | 37.38 |
| scope | 3 | 95.20 | 45.33 | 38.33 |

## Readout

- Protected shortens the LLM prefix enough to overcome its target/evidence selection overhead.
- SCOPE also shortens LLM time, but its greedy coverage selector is measured inside the online path and offsets that saving.
- These are single-sample prefill/likelihood timings, not end-to-end long-generation speedups.
