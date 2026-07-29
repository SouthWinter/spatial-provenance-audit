# SCOPE Port Parity Audit

Status: **pass**

Official source: `third_party/SCOPE/scope/clip_encoder.py` at `6bf73069e0d61307051cfda8e25925bc7b7afdd9`.

| Seed | Batch | Tokens | Width | Keep | Exact index match |
|---:|---:|---:|---:|---:|---|
| 3 | 1 | 8 | 5 | 1 | True |
| 7 | 1 | 17 | 9 | 7 | True |
| 11 | 2 | 13 | 6 | 5 | True |
| 19 | 3 | 9 | 4 | 9 | True |

## Smoke Checks

- `two_smoke_rows`: True
- `fixed_231_of_576_budget`: True
- `scope_source_recorded`: True
- `original_order_materialized`: True
- `paired_probe_mask_stable`: True

## Claim Boundary

Source-compatible official-algorithm port in the Hugging Face LLaVA backend; not a number copied from the authors' evaluation logs.
