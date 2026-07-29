# Yes/No Continuation-Length Audit

The scoring implementation masks prompt tokens and uses the model's mean cross-entropy over the remaining continuation tokens. The two candidates have equal continuation length within every model template.

| Model | yes tokens | no tokens | Equal length |
| --- | ---: | ---: | --- |
| Qwen3-VL-8B | 3 | 3 | True |
| LLaVA-1.5-7B | 4 | 4 | True |
| InternVL3.5-8B | 3 | 3 | True |

Counts include the answer token and chat-template closing tokens. Because yes and no have equal length within each backbone, candidate comparison is not affected by continuation-length imbalance.
