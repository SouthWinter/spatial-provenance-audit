# AnchorPrune Matched-Budget TextOCR-Hard Audit

All methods use LLaVA-1.5-7B, 40% visual-token retention, the same yes/no likelihood readout, and compact post-pruning positions. Differences are AnchorPrune minus the named comparator; confidence intervals use paired image-cluster bootstrap.

| Split | Method | Acc. | hFPR | PosECR | NegSRC | Correct-positive low/zero PosECR |
|---|---|---:|---:|---:|---:|---:|
| development | Full | 0.626 | 0.304 | 1.000 | 1.000 | 0.0% / 0.0% |
| development | AnchorPrune | 0.591 | 0.570 | 0.649 | 0.646 | 32.2% / 15.7% |
| development | Protected | 0.661 | 0.298 | 1.000 | 1.000 | 0.0% / 0.0% |
| development | Target | 0.623 | 0.496 | 0.458 | 0.439 | 54.7% / 30.5% |
| development | SCOPE | 0.598 | 0.508 | 0.642 | 0.642 | 32.4% / 15.1% |
| development | CoIn | 0.595 | 0.632 | 0.692 | 0.686 | 26.5% / 12.4% |
| confirmation | Full | 0.625 | 0.316 | 1.000 | 1.000 | 0.0% / 0.0% |
| confirmation | AnchorPrune | 0.578 | 0.586 | 0.635 | 0.639 | 33.2% / 18.1% |
| confirmation | Protected | 0.643 | 0.352 | 1.000 | 1.000 | 0.0% / 0.0% |
| confirmation | Random | 0.669 | 0.256 | 0.372 | 0.389 | 63.6% / 31.0% |
| confirmation | Target | 0.624 | 0.574 | 0.471 | 0.443 | 52.1% / 27.0% |
| confirmation | SCOPE | 0.586 | 0.562 | 0.614 | 0.614 | 34.3% / 16.9% |
| confirmation | CoIn | 0.579 | 0.674 | 0.666 | 0.680 | 31.0% / 14.2% |

| Confirmation comparison | Difference | 95% CI |
|---|---:|---:|
| AnchorPrune minus Full: accuracy | -0.047 | [-0.071, -0.023] |
| AnchorPrune minus Full: hfpr | +0.270 | [+0.228, +0.312] |
| AnchorPrune minus Full: positive ecr | -0.365 | [-0.402, -0.330] |
| AnchorPrune minus Full: negative source coverage | -0.361 | [-0.397, -0.326] |
| AnchorPrune minus Protected: accuracy | -0.065 | [-0.091, -0.039] |
| AnchorPrune minus Protected: hfpr | +0.234 | [+0.186, +0.282] |
| AnchorPrune minus Protected: positive ecr | -0.365 | [-0.400, -0.329] |
| AnchorPrune minus Protected: negative source coverage | -0.361 | [-0.397, -0.326] |
| AnchorPrune minus Random: accuracy | -0.091 | [-0.118, -0.064] |
| AnchorPrune minus Random: hfpr | +0.330 | [+0.284, +0.374] |
| AnchorPrune minus Random: positive ecr | +0.263 | [+0.214, +0.313] |
| AnchorPrune minus Random: negative source coverage | +0.250 | [+0.202, +0.297] |
| AnchorPrune minus Target: accuracy | -0.046 | [-0.071, -0.021] |
| AnchorPrune minus Target: hfpr | +0.012 | [-0.032, +0.058] |
| AnchorPrune minus Target: positive ecr | +0.164 | [+0.116, +0.211] |
| AnchorPrune minus Target: negative source coverage | +0.196 | [+0.146, +0.246] |
| AnchorPrune minus SCOPE: accuracy | -0.008 | [-0.023, +0.006] |
| AnchorPrune minus SCOPE: hfpr | +0.024 | [+0.000, +0.048] |
| AnchorPrune minus SCOPE: positive ecr | +0.021 | [-0.000, +0.042] |
| AnchorPrune minus SCOPE: negative source coverage | +0.025 | [+0.004, +0.046] |
| AnchorPrune minus CoIn: accuracy | -0.001 | [-0.020, +0.017] |
| AnchorPrune minus CoIn: hfpr | -0.088 | [-0.122, -0.056] |
| AnchorPrune minus CoIn: positive ecr | -0.031 | [-0.068, +0.005] |
| AnchorPrune minus CoIn: negative source coverage | -0.041 | [-0.078, -0.006] |

## Implementation and Cost

- Pinned official selector commit: `2e5d965a0e7291e46eeda73d678529d641ef74d2`.
- OpenAI CLIP ViT-L/14-336 weight SHA-256: `c6032c2e0caae3dc2d4fba35535fa6307dbb49df59c7e182b1bc4b3329b81801`.
- Exact-index parity: 12/12 deterministic tensor cases passed.
- Mean adaptive anchor size: 43.4 of 231 retained tokens.
- Mean selector time in the shared accuracy run: 670.9 ms/probe. GPU co-tenancy makes this value non-comparable; efficiency conclusions use the exclusive repeated timing report.

## Claim Boundary

This is an official-algorithm port because the upstream runtime targets the original LLaVA repository while this project evaluates the Hugging Face LLaVA checkpoint. The selector is index-identical to the pinned source on parity tests, and the adapter follows the upstream CLIP query-priority, vision-CLS attention prior, hidden-feature novelty, and native-order materialization definitions.
