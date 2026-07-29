# Locked-Confirmation Conditional Construct Audit

This cached analysis joins four Qwen masks at the same 30% visual-token budget on each of 500 locked-confirmation images. The outcome is selected-prefix minus Full-prefix yes-margin on positive probes; larger values mean better preservation of target support.

The measured-controls specification residualizes ranked variables against log evidence-box area, log median token-cell area, Full-prefix margin, and method indicators. The two-way fixed-effect specification controls every image-level attribute and every method-level shift. Keep ratio is fixed by design. Intervals resample image clusters and retain all four masks.

| Scope | Specification | Predictor | Partial rank association (95% CI) |
| --- | --- | --- | ---: |
| all_four | measured_controls_method_fe | ecr | 0.232 [0.191, 0.273] |
| all_four | measured_controls_method_fe | lpp | 0.229 [0.184, 0.273] |
| all_four | measured_controls_method_fe | geo_f1 | 0.240 [0.196, 0.283] |
| all_four | image_and_method_fe | ecr | 0.237 [0.186, 0.286] |
| all_four | image_and_method_fe | lpp | 0.219 [0.167, 0.271] |
| all_four | image_and_method_fe | geo_f1 | 0.232 [0.181, 0.281] |
| deletion_only | measured_controls_method_fe | ecr | 0.279 [0.230, 0.328] |
| deletion_only | measured_controls_method_fe | lpp | 0.257 [0.206, 0.309] |
| deletion_only | measured_controls_method_fe | geo_f1 | 0.268 [0.219, 0.316] |
| deletion_only | image_and_method_fe | ecr | 0.253 [0.188, 0.315] |
| deletion_only | image_and_method_fe | lpp | 0.224 [0.158, 0.289] |
| deletion_only | image_and_method_fe | geo_f1 | 0.235 [0.169, 0.299] |

These are convergent-validity diagnostics, not causal estimates: token-origin geometry cannot establish what information a contextualized token carries or what the decoder uses.
