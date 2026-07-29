# Human-Validated Edit-Region Audit

The region is the renderer-derived glyph box at the unique substituted-character position. All 102 edits passed human semantic QC.

| Method | n | Accuracy | Word ECR | Edit ECR | Edit ECR=0 | Word>=0.5, Edit<0.5 |
|---|---:|---:|---:|---:|---:|---:|
| Full | 102 | 0.873 | 1.000 | 1.000 | 0.000 | 0.000 |
| Target | 102 | 0.912 | 0.869 | 0.897 | 0.029 | 0.059 |
| Random | 102 | 0.716 | 0.304 | 0.311 | 0.578 | 0.029 |
| Grid | 102 | 0.588 | 0.386 | 0.411 | 0.363 | 0.088 |

| Comparison | Accuracy diff [95% CI] | Edit-ECR diff [95% CI] | Word-ECR diff [95% CI] |
|---|---:|---:|---:|
| Target - Random | +0.196 [+0.108, +0.284] | +0.586 [+0.492, +0.676] | +0.565 [+0.488, +0.638] |
| Target - Grid | +0.324 [+0.225, +0.422] | +0.486 [+0.397, +0.575] | +0.482 [+0.414, +0.552] |
