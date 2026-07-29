# Repeated Exclusive AnchorPrune Timing

Three fresh-process repetitions use the same first 100 image-disjoint TextOCR-Hard confirmation probes, one A800 GPU, float16, eager attention, and rotated method order. The first 5 probes of each process are discarded as warm-up. Selector cost is included online.

| Method | Keep | Vision ms | Select/materialize ms | LLM ms | Total ms | Speedup |
|---|---:|---:|---:|---:|---:|---:|
| full | 1.000 | 9.9 +/- 0.2 | 0.9 +/- 0.0 | 63.6 +/- 0.1 | 75.4 +/- 0.3 | 1.00x |
| target | 0.401 | 9.9 +/- 0.4 | 5.2 +/- 0.1 | 37.9 +/- 0.3 | 53.9 +/- 0.9 | 1.40x |
| anchorprune | 0.401 | 10.0 +/- 0.0 | 13.4 +/- 0.1 | 37.8 +/- 0.0 | 62.0 +/- 0.1 | 1.22x |

Values are mean +/- sample standard deviation across three repetitions; each repetition averages 95 post-warm-up probes. These are single-sample likelihood/prefill timings, not end-to-end long-generation speedups.
