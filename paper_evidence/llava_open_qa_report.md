# LLaVA Open-OCR-QA Paired Audit

Original questions are used for greedy generation. The selector receives only the question, never the gold answer or evidence boxes.

| Task | n | Metric | Full | Target (40%) | Delta | 95% CI | p | Win/Loss/Tie | Keep |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| TextVQA-lite | 500 | textvqa_accuracy | 0.4950 | 0.2710 | -0.2240 | [-0.2636, -0.1850] | 0.000050 | 15/141/344 | 0.4010 |
| DocVQA-lite | 500 | anls | 0.2551 | 0.1358 | -0.1194 | [-0.1502, -0.0888] | 0.000050 | 16/96/388 | 0.4010 |

The intervals exclude zero for both tasks. At this aggressive budget, question-conditioned Target pruning is therefore not backbone-robust for native open-answer generation.
