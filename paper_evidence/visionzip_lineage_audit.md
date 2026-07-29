# VisionZip Merge-Lineage Audit

VisionZip retains dominant tokens and exhaustively assigns every remaining source token to a contextual output token. LineageECR therefore uses every contributing source cell; AnchorECR uses only representative output locations.

| Model | Method | n | Full tokens | Output tokens | Contextual outputs | Lineage PosECR | Anchor PosECR |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3-VL-8B | VisionZip (30%), confirmation | 1000 | 794.2 | 238.7 | 37.5 | 1.000 | 0.846 |
| LLaVA-1.5-7B | VisionZip (40%), confirmation | 1000 | 576.0 | 231.0 | 36.0 | 1.000 | 0.563 |
