# Beyond Accuracy: Spatial-Provenance Audit

This repository accompanies **Beyond Accuracy: Auditing Spatial Provenance in
Visual Token Pruning for OCR-Critical MLLM Inference** by Feixiang Liu, Qiang
Qiu, Hao Zhang, and Xinyue Wang.

**Paper:** [arXiv:2608.00077](https://arxiv.org/abs/2608.00077)

It contains the code and compact audit outputs needed to reconstruct
the paper's TextOCR-Hard protocol, run the three model backends, reproduce
matched-budget selectors, and rebuild statistical summaries. It intentionally
does not redistribute model checkpoints, benchmark images, or source-dataset
annotations.

## Environment

The reported runs used one NVIDIA A800 80GB GPU, CUDA 12.4, PyTorch 2.6.0,
Transformers 4.57.1, and eager attention where attention weights were required.
Create the tested software environment with:

```bash
conda env create -f environment.yml
conda activate spatial-provenance-audit
pytest -q
```

Model checkpoints are supplied through command-line arguments. The paper uses:

- `Qwen/Qwen3-VL-8B-Instruct` or an equivalent local snapshot;
- `llava-hf/llava-1.5-7b-hf`;
- `OpenGVLab/InternVL3_5-8B-HF`.

## Build TextOCR-Hard

Obtain TextOCR images and annotations under the source dataset's terms, then
convert the word annotations and construct deterministic hard probes:

```bash
python -m recap.cli prepare-textocr-regions \
  --annotation-file /path/to/TextOCR_0.1_val.json \
  --image-root /path/to/textocr/images \
  --output data/textocr_val_regions.jsonl

python scripts/build_textocr_hard_probes.py \
  --input data/textocr_val_regions.jsonl \
  --output data/textocr_val_hard_probes_500img.jsonl \
  --limit-images 500 --seed 17 \
  --construction-version development-v1 \
  --expected-sha256 e50c20d2b4de5ba6f76a96087c2297a9d6e7deb114884df29d8baa5097c1d3f9

python scripts/build_textocr_hard_probes.py \
  --input data/textocr_val_regions.jsonl \
  --output data/textocr_val_hard_confirmation_500img_seed20260720.jsonl \
  --limit-images 500 --seed 20260720 --shuffle-images \
  --construction-version confirmation-v2 \
  --exclude-probes data/textocr_val_hard_probes_500img.jsonl \
  --expected-sha256 0ad15992e0a18e4568e44a78de7177aef6e401ce075a5cd7e2506764d72a6aa8
```

The development set predates a Unicode-safety correction in decoy generation,
so the two locked sets name their construction versions explicitly. The
versions differ only in candidate generation for non-ASCII letters and digits;
the SHA assertions prevent silent protocol drift. See
`LOCKED_TEXTOCR_HARD_PROTOCOL.md` for the complete lock record. Audit the
generated negative set before model execution:

```bash
python scripts/build_hard_negative_lexical_audit.py
python scripts/build_hard_negative_human_qc_launch.py
```

## Run Main Selectors

The commands below show the fixed paper operating points. Replace checkpoint
arguments and output directories as needed.

```bash
# Qwen box-free Target, 30% visual-token retention
python -m recap.cli run-qwen-pruned \
  --input data/textocr_val_hard_probes_500img.jsonl --is-probes \
  --work-dir runs/qwen_target_0p30 \
  --pretrained /path/to/Qwen3-VL-8B-Instruct \
  --selector target_embed_topk --keep-ratio 0.30 \
  --embedding-relevance-weight 0.85 --embedding-query-topk 2 \
  --min-pixels 802816 --max-pixels 802816 \
  --device cuda --device-map auto --attn-implementation eager

# LLaVA box-assisted Protected, 40%
python -m recap.cli run-llava-pruned \
  --input data/textocr_val_hard_probes_500img.jsonl --is-probes \
  --work-dir runs/llava_protected_0p40 \
  --pretrained llava-hf/llava-1.5-7b-hf \
  --selector target_embed_protected_topk --keep-ratio 0.40 \
  --device cuda --device-map auto --dtype bfloat16

# InternVL Soft evidence, 50%; calibrate thresholds on the grouped dev split
python -m recap.cli run-internvl-pruned \
  --input data/textocr_val_hard_probes_500img.jsonl --is-probes \
  --work-dir runs/internvl_soft_0p50 \
  --pretrained OpenGVLab/InternVL3_5-8B-HF \
  --selector target_embed_soft_evidence_topk --keep-ratio 0.50 \
  --evidence-boost 0.05 --device cuda --device-map auto

python scripts/calibrate_yesno_thresholds.py --help
```

Run `python -m recap.cli <command> --help` for every backend option. Each pruned
run writes probe scores, sample scores, metrics, and per-probe pruning traces.
The LLaVA development row in the main table uses bfloat16 as shown above. The
separately locked confirmation script retains its original float16 protocol;
dtype is therefore part of each reported run's provenance and is not mixed
within a comparison.

## Image-Dependence Controls

Build same-sized gray images and three collision-free wrong-image derangements
from the locked probe file and the full TextOCR region index:

```bash
python scripts/build_textocr_image_free_controls.py \
  --input data/textocr_val_hard_confirmation_500img_seed20260720.jsonl \
  --regions data/textocr_val_regions_full.jsonl \
  --output-dir data/textocr_image_free_controls/confirmation \
  --seeds 101,202,303
```

Positive and negative probes from one source image share the same control image.
The constructor rejects fixed points and target/source collisions against every
OCR token in the wrong image under seven normalizers, and writes hashes plus the
complete permutation manifest. Run each backbone with its checkpoint variable:

```bash
MODEL=qwen QWEN3_PATH=/path/to/Qwen3-VL-8B-Instruct \
  INPUT_DIR=data/textocr_image_free_controls/confirmation JOBS=1 \
  bash scripts/run_textocr_image_free_controls.sh

MODEL=llava LLAVA_PATH=llava-hf/llava-1.5-7b-hf \
  INPUT_DIR=data/textocr_image_free_controls/confirmation JOBS=1 \
  bash scripts/run_textocr_image_free_controls.sh
```

For InternVL, first construct the controls from the development probe file and
filter them with the unchanged image-group hash split:

```bash
python scripts/build_textocr_image_free_controls.py \
  --input data/textocr_val_hard_probes_500img.jsonl \
  --regions data/textocr_val_regions_full.jsonl \
  --output-dir data/textocr_image_free_controls/development \
  --seeds 101,202,303
python scripts/filter_textocr_image_free_controls.py \
  --input-dir data/textocr_image_free_controls/development \
  --output-dir data/textocr_image_free_controls/development_test536 \
  --split test
MODEL=internvl INTERNVL_PATH=OpenGVLab/InternVL3_5-8B-HF \
  INPUT_DIR=data/textocr_image_free_controls/development_test536 JOBS=1 \
  bash scripts/run_textocr_image_free_controls.sh
```

Place or link the matched full-prefix score files under the corresponding model
directories, then reproduce Table 2 and Supplementary Tables S18--S19 with:

```bash
python scripts/build_textocr_image_free_report.py \
  --run-root runs/problem_optimization_audit/image_free_controls \
  --output-dir runs/problem_optimization_audit/image_free_controls/report
```

The packaged aggregate report records every per-seed interval and the stricter
subset whose decoy is a genuine OCR token elsewhere in TextOCR. It contains no
benchmark images or row-level source annotations.

For the LLaVA original-question 500-sample TextVQA/DocVQA boundary experiment,
run the question-only selector and then build the paired report. The internal
`*_lite` task identifier denotes this fixed diagnostic split; it is not a new
benchmark:

```bash
python scripts/run_llava_open_ocr_qa_generation.py \
  --task textvqa_val_lite --work-dir runs/llava_textvqa_target_0p40 \
  --pretrained llava-hf/llava-1.5-7b-hf \
  --selector target_embed_topk --keep-ratio 0.40 --limit 500 \
  --dtype float16 --attn-implementation eager

python scripts/build_llava_open_ocr_qa_report.py --help
```

For the full-validation matched-budget control, set `QWEN_PATH` and
`LLAVA_PATH`, then run:

```bash
QWEN_PATH=Qwen/Qwen3-VL-8B-Instruct \
LLAVA_PATH=llava-hf/llava-1.5-7b-hf \
PYTHON=python bash scripts/run_open_qa_matched_baselines.sh

python scripts/build_full_open_ocr_qa_report.py \
  --root runs/open_ocr_qa_full \
  --output-dir runs/open_ocr_qa_full/report
```

The runner reuses the locked Full-prefix answers and evaluates Random and Grid
at the same 70% visual-token budget as Target/Target-Grid on all 5000 TextVQA
and 5349 DocVQA validation examples. The report verifies identical sample IDs
and Full answers before producing paired bootstrap intervals.

To reproduce the construct-validity analysis that separates geometry from
model confidence, run:

```bash
python scripts/build_ecr_construct_validity_audit.py --bootstrap 10000
python scripts/build_confirmation_conditional_construct_audit.py
```

The first report contains intervention-based marginal and partial Spearman
correlations. The second joins four Qwen masks at the same 30% locked-
confirmation budget and controls evidence area, token-cell area, Full-prefix
margin, method identity, and image identity. Both are construct diagnostics,
not causal estimates.

To repeat the locked-confirmation random-mask audit, run the five additional
fixed masks and then combine them with the original seed:

```bash
bash scripts/run_qwen_confirmation_random_seeds.sh
python scripts/build_confirmation_random_seed_stats.py
```

## Recent External Baselines

The external rows use three distinct evidence levels:

| Method | Public source revision | Evidence level | Reproduction entry point |
| --- | --- | --- | --- |
| FastV | `pkunlp-icler/fastv@d1659729b5bf1be225e99ee15783deeea80f63b1` | LLaVA official-algorithm port; 20--50% curve | `build_official_baseline_extension_report.py` |
| VisionZip | `JIA-Lab-research/VisionZip@8f86b55c6f000eb033e6912538af2dd7dcb30502` | LLaVA official-algorithm port and scoped Qwen 30% native port | `build_visionzip_lineage_audit.py` |
| SCOPE | `kinredon/SCOPE@6bf73069e0d61307051cfda8e25925bc7b7afdd9` | public selector; exact-index parity on independent tensors | `audit_scope_port_parity.py` |
| AnchorPrune | `MULTI-cau/AnchorPrune@2e5d965a0e7291e46eeda73d678529d641ef74d2` | public selector; exact-index parity on 12 deterministic cases | `test_anchorprune_port.py` |
| CoIn | published algorithm; no public code linked at evaluation time | independently written paper-algorithm port, no code-parity claim | `build_coin_textocr_audit.py` |

Qwen FastV and InternVL FastV/VisionZip are not presented as official ports.
Any embedding-salience rows for those backbones remain explicitly labeled as
protocol proxies or unsupported.

The LLaVA SCOPE port follows the public selector at commit
`6bf73069e0d61307051cfda8e25925bc7b7afdd9`, using the official default
`alpha=1`. Clone that revision beside the artifact, then verify exact selected
indices before running the 40% comparison:

```bash
git clone https://github.com/kinredon/SCOPE.git third_party/SCOPE
git -C third_party/SCOPE checkout 6bf73069e0d61307051cfda8e25925bc7b7afdd9
python scripts/audit_scope_port_parity.py

python -m recap.cli run-llava-pruned \
  --input data/textocr_val_hard_probes_500img.jsonl --is-probes \
  --work-dir runs/llava_scope_0p40 \
  --pretrained llava-hf/llava-1.5-7b-hf \
  --selector scope --keep-ratio 0.40 \
  --device cuda --device-map auto --dtype float16 \
  --attn-implementation eager
```

To isolate the same selector's facility-location coverage component, rerun it
with all saliency weights set to one, then build the paired image-cluster audit:

```bash
python -m recap.cli run-llava-pruned \
  --input data/textocr_val_hard_confirmation_500img_seed20260720.jsonl --is-probes \
  --work-dir runs/llava_scope_pure_coverage_0p40 \
  --pretrained llava-hf/llava-1.5-7b-hf \
  --selector scope --scope-alpha 0 --keep-ratio 0.40 \
  --device cuda --device-map auto --dtype float16 \
  --attn-implementation eager

python scripts/build_scope_coverage_ablation.py --help
```

This is the same SCOPE selector endpoint used by the concurrent calibration
study cited in the paper, but it is evaluated here on TextOCR-Hard with a
sequence-likelihood readout. The packaged report keeps confidence calibration,
answer behavior, positive spatial provenance, and negative-source coverage
separate under the locked confirmation protocol.

The LLaVA AnchorPrune comparison follows the official Apache-2.0 selector at
commit `2e5d965a0e7291e46eeda73d678529d641ef74d2`. The adapter changes only the
model interface for the Hugging Face checkpoint; CLIP query priority,
CLS-attention importance, two-stage novelty expansion, and native-order
materialization follow upstream. Clone the pinned source and point the run to
an OpenAI CLIP ViT-L/14-336 checkpoint:

```bash
git clone https://github.com/MULTI-cau/AnchorPrune.git third_party/AnchorPrune
git -C third_party/AnchorPrune checkout 2e5d965a0e7291e46eeda73d678529d641ef74d2
export ANCHORPRUNE_CLIP_PATH=/path/to/clip-vit-large-patch14-336
pytest -q tests/test_anchorprune_port.py
bash scripts/run_anchorprune_textocr.sh
python scripts/build_anchorprune_textocr_report.py
```

The parity test checks exact anchor and final selected indices on 12
deterministic cases. Model checkpoints and CLIP weights are excluded from the
artifact.

The CoIn comparison is an independently written paper-algorithm port because no
public implementation was linked at evaluation time. It follows the published
incremental coverage procedure and reported LLaVA-1.5 hyperparameters:

```bash
python -m recap.cli run-llava-pruned \
  --input data/textocr_val_hard_probes_500img.jsonl --is-probes \
  --work-dir runs/llava_coin_0p40 \
  --pretrained llava-hf/llava-1.5-7b-hf \
  --selector coin --keep-ratio 0.40 \
  --coin-alpha 0.90 --coin-beta 0.60 \
  --device cuda --device-map auto --dtype float16 \
  --attn-implementation eager
```

## Statistical and Timing Audits

```bash
python scripts/paired_prune_stats.py --help
python scripts/build_p0_stats_and_failures.py
python scripts/benchmark_hf_prune_batch_prefill.py --help
bash scripts/run_coin_repeated_timing.sh
bash scripts/run_anchorprune_repeated_timing.sh
python scripts/build_scope_textocr_audit.py
python scripts/build_coin_textocr_audit.py
python scripts/build_correct_positive_provenance_audit.py
python scripts/build_scope_coverage_ablation.py --help
```

## Position-ID Replay

The artifact includes the exact main-row LLaVA Protected and InternVL Soft
evidence masks under `paper_evidence/position_id/`. Replay either file under
compact and preserved logical positions while holding physical token presence,
order, prompts, model weights, and bfloat16 inference fixed:

```bash
python -m recap.cli run-llava-pruned \
  --input data/textocr_val_hard_probes_500img.jsonl --is-probes \
  --work-dir runs/llava_protected40_preserve \
  --pretrained llava-hf/llava-1.5-7b-hf \
  --selector target_embed_protected_topk --keep-ratio 0.40 \
  --kept-indices paper_evidence/position_id/llava_protected40_main_masks.jsonl \
  --position-mode preserve --dtype bfloat16 \
  --device cuda --device-map auto --attn-implementation eager

python -m recap.cli run-internvl-pruned \
  --input data/textocr_val_hard_probes_500img.jsonl --is-probes \
  --work-dir runs/internvl_soft50_preserve \
  --pretrained OpenGVLab/InternVL3_5-8B-HF \
  --selector target_embed_soft_evidence_topk --keep-ratio 0.50 \
  --evidence-boost 0.05 \
  --kept-indices paper_evidence/position_id/internvl_soft50_main_masks.jsonl \
  --position-mode preserve --dtype bfloat16 \
  --device cuda --device-map auto --attn-implementation eager
```

Run the same commands with `--position-mode compact`, then compare score files
with `scripts/analyze_position_id_policy.py`. The packaged alignment report and
policy summaries record zero sample-ID, kept-index, and ECR mismatches. Mask
SHA256 values are also listed in `position_id_main_row_alignment.md`.

The `paper_evidence/` directory contains compact, non-image audit summaries,
the correct-positive spatial-provenance and conditional-correlation audits,
full-validation open-QA matched controls, exact position-replay masks,
merge-aware VisionZip lineage and anchor audits, seed-integrated random-control
uncertainty, a human-validated substituted-glyph coverage audit, a Qwen/LLaVA
threshold-free and shared-threshold calibration decomposition, SCOPE/CoIn
development and confirmation statistics, and the five-method three-repetition
timing readout used by the manuscript. See
`DATA_AND_ANNOTATION.md` for annotation provenance and
`RELEASE_AND_LICENSE_MATRIX.md` for component-level redistribution boundaries.

## Citation

```bibtex
@article{liu2026beyond,
  title   = {Beyond Accuracy: Auditing Spatial Provenance in Visual Token Pruning for OCR-Critical MLLM Inference},
  author  = {Liu, Feixiang and Qiu, Qiang and Zhang, Hao and Wang, Xinyue},
  journal = {arXiv preprint arXiv:2608.00077},
  year    = {2026}
}
```

## License

Author-generated code in this repository is released under the Apache License
2.0. Third-party implementations, model checkpoints, and datasets remain
subject to their respective licenses and terms. See
`RELEASE_AND_LICENSE_MATRIX.md` and
`third_party_licenses/ANCHORPRUNE_LICENSE.txt` for details.
