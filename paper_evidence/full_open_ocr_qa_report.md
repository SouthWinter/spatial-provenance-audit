# Full-Validation Open OCR/Document QA

All selectors receive the original question only. Confidence intervals are paired sample bootstrap intervals.

| Model | Task | Method | Budget | n | Metric | Full | Pruned | Delta | 95% CI | Win/Loss/Tie | Keep |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| Qwen3-VL-8B | TextVQA | Target-Grid | 30% | 5000 | TextVQA accuracy | 0.8282 | 0.6346 | -0.1936 | [-0.2057, -0.1816] | 183/1269/3548 | 0.3000 |
| Qwen3-VL-8B | TextVQA | Target-Grid | 70% | 5000 | TextVQA accuracy | 0.8282 | 0.7947 | -0.0335 | [-0.0403, -0.0270] | 122/308/4570 | 0.7000 |
| Qwen3-VL-8B | DocVQA | Target-Grid | 30% | 5349 | ANLS | 0.9395 | 0.5291 | -0.4103 | [-0.4228, -0.3976] | 91/3101/2157 | 0.3000 |
| Qwen3-VL-8B | DocVQA | Target-Grid | 70% | 5349 | ANLS | 0.9395 | 0.8462 | -0.0933 | [-0.1015, -0.0855] | 94/944/4311 | 0.7000 |
| LLaVA-1.5-7B | TextVQA | Target | 40% | 5000 | TextVQA accuracy | 0.4847 | 0.2766 | -0.2081 | [-0.2203, -0.1958] | 150/1279/3571 | 0.4010 |
| LLaVA-1.5-7B | TextVQA | Target | 70% | 5000 | TextVQA accuracy | 0.4847 | 0.3588 | -0.1259 | [-0.1364, -0.1155] | 139/832/4029 | 0.7014 |
| LLaVA-1.5-7B | DocVQA | Target | 40% | 5349 | ANLS | 0.2157 | 0.1181 | -0.0976 | [-0.1063, -0.0889] | 183/842/4324 | 0.4010 |
| LLaVA-1.5-7B | DocVQA | Target | 70% | 5349 | ANLS | 0.2157 | 0.1619 | -0.0538 | [-0.0616, -0.0461] | 221/618/4510 | 0.7014 |
| Qwen3-VL-8B | TextVQA | Random | 70% | 5000 | TextVQA accuracy | 0.8282 | 0.8102 | -0.0180 | [-0.0235, -0.0125] | 98/210/4692 | 0.7000 |
| Qwen3-VL-8B | TextVQA | Grid | 70% | 5000 | TextVQA accuracy | 0.8282 | 0.7957 | -0.0325 | [-0.0390, -0.0261] | 110/289/4601 | 0.7000 |
| Qwen3-VL-8B | DocVQA | Random | 70% | 5349 | ANLS | 0.9395 | 0.8604 | -0.0791 | [-0.0866, -0.0717] | 89/829/4431 | 0.7000 |
| Qwen3-VL-8B | DocVQA | Grid | 70% | 5349 | ANLS | 0.9395 | 0.8218 | -0.1177 | [-0.1268, -0.1087] | 81/913/4355 | 0.7000 |
| LLaVA-1.5-7B | TextVQA | Random | 70% | 5000 | TextVQA accuracy | 0.4847 | 0.4491 | -0.0357 | [-0.0434, -0.0281] | 163/364/4473 | 0.7014 |
| LLaVA-1.5-7B | TextVQA | Grid | 70% | 5000 | TextVQA accuracy | 0.4847 | 0.4382 | -0.0465 | [-0.0544, -0.0388] | 142/397/4461 | 0.7014 |
| LLaVA-1.5-7B | DocVQA | Random | 70% | 5349 | ANLS | 0.2157 | 0.1819 | -0.0338 | [-0.0406, -0.0270] | 210/498/4641 | 0.7014 |
| LLaVA-1.5-7B | DocVQA | Grid | 70% | 5349 | ANLS | 0.2157 | 0.1893 | -0.0264 | [-0.0327, -0.0200] | 208/413/4728 | 0.7014 |

| Model | Task | Comparison | Recovery | 95% CI | Improved/Degraded/Tied |
|---|---|---|---:|---:|---:|
| Qwen3-VL-8B | TextVQA | 70% minus 30% | +0.1601 | [+0.1483, +0.1719] | 1128/226/3646 |
| Qwen3-VL-8B | DocVQA | 70% minus 30% | +0.3170 | [+0.3038, +0.3300] | 2691/305/2353 |
| LLaVA-1.5-7B | TextVQA | 70% minus 40% | +0.0822 | [+0.0732, +0.0913] | 568/121/4311 |
| LLaVA-1.5-7B | DocVQA | 70% minus 40% | +0.0438 | [+0.0376, +0.0502] | 453/135/4761 |

| Model | Task | Budget | Comparison | Delta | 95% CI | Win/Loss/Tie | Target/Base keep |
|---|---|---:|---|---:|---:|---:|---:|
| Qwen3-VL-8B | TextVQA | 70% | Target-Grid minus Random | -0.0156 | [-0.0231, -0.0081] | 236/307/4457 | 0.7000/0.7000 |
| Qwen3-VL-8B | TextVQA | 70% | Target-Grid minus Grid | -0.0011 | [-0.0086, +0.0065] | 277/286/4437 | 0.7000/0.7000 |
| Qwen3-VL-8B | DocVQA | 70% | Target-Grid minus Random | -0.0142 | [-0.0239, -0.0046] | 671/771/3907 | 0.7000/0.7000 |
| Qwen3-VL-8B | DocVQA | 70% | Target-Grid minus Grid | +0.0244 | [+0.0131, +0.0355] | 807/776/3766 | 0.7000/0.7000 |
| LLaVA-1.5-7B | TextVQA | 70% | Target minus Random | -0.0902 | [-0.1011, -0.0796] | 243/731/4026 | 0.7014/0.7014 |
| LLaVA-1.5-7B | TextVQA | 70% | Target minus Grid | -0.0793 | [-0.0904, -0.0685] | 281/721/3998 | 0.7014/0.7014 |
| LLaVA-1.5-7B | DocVQA | 70% | Target minus Random | -0.0200 | [-0.0276, -0.0123] | 356/505/4488 | 0.7014/0.7014 |
| LLaVA-1.5-7B | DocVQA | 70% | Target minus Grid | -0.0274 | [-0.0353, -0.0195] | 341/537/4471 | 0.7014/0.7014 |
