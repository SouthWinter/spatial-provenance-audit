# Correct-Positive Spatial-Provenance Audit

This locked-confirmation audit conditions on positive probes answered correctly. ECR is geometric overlap between retained token cells and the annotated positive region; it is not an information-theoretic measure of whether contextualized tokens encode the region.

| Method | Correct positive | Mean lineage ECR | Mean anchor ECR | ECR < 0.50 | ECR = 0 |
|---|---:|---:|---:|---:|---:|
| Qwen Target (30%) | 406 | 0.672 | 0.672 | 113 (27.8%; 23.7--32.4) | 53 (13.1%; 10.1--16.7) |
| Qwen Random (30%) | 319 | 0.289 | 0.289 | 227 (71.2%; 66.0--75.9) | 118 (37.0%; 31.9--42.4) |
| Qwen Grid (30%) | 294 | 0.371 | 0.371 | 192 (65.3%; 59.7--70.5) | 61 (20.7%; 16.5--25.7) |
| LLaVA Protected (40%) | 319 | 1.000 | 1.000 | 0 (0.0%; 0.0--1.2) | 0 (0.0%; 0.0--1.2) |
| LLaVA Random (40%) | 297 | 0.374 | 0.374 | 189 (63.6%; 58.0--68.9) | 92 (31.0%; 26.0--36.5) |
| LLaVA Target (40%) | 411 | 0.486 | 0.486 | 214 (52.1%; 47.2--56.9) | 111 (27.0%; 22.9--31.5) |
| LLaVA SCOPE (40%) | 367 | 0.637 | 0.637 | 126 (34.3%; 29.7--39.3) | 62 (16.9%; 13.4--21.1) |
| LLaVA SCOPE pure coverage (40%) | 403 | 0.626 | 0.626 | 143 (35.5%; 31.0--40.3) | 63 (15.6%; 12.4--19.5) |
| LLaVA CoIn (40%) | 416 | 0.677 | 0.677 | 129 (31.0%; 26.8--35.6) | 59 (14.2%; 11.2--17.9) |
| LLaVA VisionZip (40%) | 417 | 1.000 | 0.576 | 0 (0.0%; 0.0--0.9) | 0 (0.0%; 0.0--0.9) |
