# Locked TextOCR-Hard Construction Protocol

## Data Contract

- Development set: 500 TextOCR validation images and 1,000 paired probes.
- Development SHA-256: `e50c20d2b4de5ba6f76a96087c2297a9d6e7deb114884df29d8baa5097c1d3f9`.
- Confirmation set: 500 different TextOCR validation images and 1,000 paired
  probes, sampled after excluding every development image identifier.
- Confirmation SHA-256: `0ad15992e0a18e4568e44a78de7177aef6e401ce075a5cd7e2506764d72a6aa8`.
- Image overlap between the two sets: zero.

## Versioned Construction

The development split was created with seed 17 under `development-v1`. Before
the confirmation split was sampled, decoy candidate handling was made safe for
non-ASCII letters and digits. The confirmation split was then locked with seed
20260720 under `confirmation-v2` and deterministic source-image shuffling.
Both versions keep the same eligibility filters, token sampling, paired-probe
schema, and image-disjoint confirmation contract. Versioning preserves the
already evaluated development set instead of silently replacing it.

The exact commands and mandatory output-hash checks appear in `README.md`.

## Locked Operating Points

| Backbone | Role | Selector | Keep ratio | Seed |
|---|---|---|---:|---:|
| Qwen3-VL-8B | full prefix | top-k/full | 1.00 | 17 |
| Qwen3-VL-8B | proposed | target-conditioned top-k | 0.30 | 17 |
| Qwen3-VL-8B | matched random | random | 0.30 | 17 |
| Qwen3-VL-8B | matched grid | grid | 0.30 | 17 |
| LLaVA-1.5-7B | full prefix | direct inference | 1.00 | 17 |
| LLaVA-1.5-7B | proposed | protected top-k | 0.40 | 17 |
| LLaVA-1.5-7B | matched random | random | 0.40 | 17 |
| LLaVA-1.5-7B | external baseline | VisionZip | 0.40 | 17 |

No confirmation result was used to change these selectors, keep ratios, seeds,
or decoding rules. Later SCOPE and CoIn audits use the same frozen 40% LLaVA
budget and are labeled separately in the paper.
