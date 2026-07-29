"""Command line entrypoint for standalone RECAP."""

from __future__ import annotations

import argparse
from pathlib import Path

from recap.aggregate import aggregate_scores
from recap.audit import audit_samples
from recap.gsrbench import GSRBENCH_CONFIGS, GSRBENCH_GROUPS, prepare_gsrbench
from recap.images import validate_probe_images
from recap.io import read_json, read_jsonl, write_json, write_jsonl
from recap.probes import PROBE_MODES, build_probe_dataset
from recap.recap_ablation import DEFAULT_RISKS as DEFAULT_RECAP_ABLATION_RISKS
from recap.recap_ablation import ablate_recap_probe_scores
from recap.recap_analysis import (
    LOW_COST_VARIANTS,
    bootstrap_ci_to_csv_rows,
    coverage_curves_to_csv_rows,
    extract_recap_case_studies,
    filter_recap_probes,
    low_cost_equivalence_to_csv_rows,
    recap_bootstrap_ci,
    recap_cost_utility,
    recap_coverage_curves,
    recap_low_cost_equivalence,
    write_case_jsonl,
    write_csv,
)
from recap.reporting import compact_metrics
from recap.relations import RELATION_FAMILIES, normalize_relation
from recap.vsr import prepare_vsr
from recap.whatsup import WHATUP_CONFIGS, WHATUP_GROUPS, prepare_whatsup


def _csv_items(value: str) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def _parse_relations(value: str) -> set[str] | None:
    items = {normalize_relation(item) for item in _csv_items(value)}
    items.discard("")
    return items or None


def _parse_relation_families(value: str) -> set[str] | None:
    items = {item.lower().replace("-", "_").strip() for item in _csv_items(value)}
    unknown = sorted(item for item in items if item not in RELATION_FAMILIES)
    if unknown:
        raise ValueError(f"Unknown relation families {unknown}. Use one of: {sorted(RELATION_FAMILIES)}")
    return items or None


def _parse_grid_shape(value: str) -> tuple[int, int | None]:
    text = str(value or "24").strip().lower().replace("*", "x")
    if "x" not in text:
        rows = int(text)
        return rows, None
    left, right = [part.strip() for part in text.split("x", 1)]
    return int(left), int(right)


def _load_ranked_risk_scores(path: str, key: str) -> dict[str, float]:
    from recap.metrics import rank_norm

    rows = read_jsonl(path)
    sample_ids: list[str] = []
    values: list[float] = []
    for row in rows:
        if key not in row:
            continue
        sample_id = str(row.get("sample_id", row.get("id", "")))
        if not sample_id:
            continue
        sample_ids.append(sample_id)
        values.append(float(row[key]))
    if not sample_ids:
        raise ValueError(f"No usable risk scores found in {path!r} for key {key!r}.")
    return dict(zip(sample_ids, rank_norm(values)))


def _load_budget_ratios(path: str, key: str = "keep_ratio") -> dict[str, float]:
    rows = read_jsonl(path)
    ratios: dict[str, float] = {}
    for row in rows:
        sample_id = str(row.get("sample_id", row.get("id", "")))
        if not sample_id or key not in row:
            continue
        ratio = float(row[key])
        if ratio <= 0.0 or ratio > 1.0:
            raise ValueError(f"Budget ratio for {sample_id!r} must be in (0, 1], got {ratio}.")
        ratios[sample_id] = ratio
    if not ratios:
        raise ValueError(f"No usable budget ratios found in {path!r} for key {key!r}.")
    return ratios


def _load_kept_indices(path: str, key: str = "kept_indices") -> dict[str, list[int]]:
    rows = read_jsonl(path)
    out: dict[str, list[int]] = {}
    for row in rows:
        sample_id = str(row.get("sample_id", row.get("id", "")))
        raw_indices = row.get(key)
        if not sample_id or not isinstance(raw_indices, list):
            continue
        out[sample_id] = [int(idx) for idx in raw_indices]
    if not out:
        raise ValueError(f"No usable kept-index rows found in {path!r} for key {key!r}.")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone RECAP experiment pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_vsr = subparsers.add_parser("prepare-vsr", help="Download/process VSR and convert it to RECAP canonical JSONL.")
    p_vsr.add_argument("--split", default="test")
    p_vsr.add_argument("--dataset-path", default="random", help="HF dataset path or shortcut: random, zeroshot.")
    p_vsr.add_argument("--metadata-file", default="", help="Local VSR metadata file (.jsonl/.json/.csv/.parquet), bypassing Hugging Face.")
    p_vsr.add_argument("--coco-root", default="", help="Local COCO root containing train2017/val2017 images.")
    p_vsr.add_argument("--image-dir", default="", help="Optional directory to copy/download selected VSR images into.")
    p_vsr.add_argument("--output", required=True)
    p_vsr.add_argument("--cache-dir", default=None)
    p_vsr.add_argument("--hf-endpoint", default="", help="HF Hub endpoint, for example https://hf-mirror.com.")
    p_vsr.add_argument("--local-files-only", action="store_true", help="Use Hugging Face local cache only; do not attempt network access.")
    p_vsr.add_argument("--metadata-only", action="store_true", help="Do not copy/download images; write image filenames or URLs only.")
    p_vsr.add_argument("--allow-missing-images", action="store_true", help="Allow unresolved VSR image paths; useful only for non-visual debugging.")
    p_vsr.add_argument("--keep-non-left-right", action="store_true", help="Keep all spatial relations for controls; default keeps only left/right.")
    p_vsr.add_argument("--relations", default="", help="Comma-separated normalized relations to keep, for example above,below,inside,contains.")
    p_vsr.add_argument("--relation-families", default="", help=f"Comma-separated relation families to keep. Choices: {','.join(sorted(RELATION_FAMILIES))}.")
    p_vsr.add_argument("--limit", type=int, default=None)

    whatsup_choices = sorted([*WHATUP_CONFIGS, *WHATUP_GROUPS])
    p_whatsup = subparsers.add_parser("prepare-whatsup", help="Download/process WhatsUp/What'sUp and convert it to RECAP canonical JSONL.")
    p_whatsup.add_argument("--dataset", default="controlled_a", choices=whatsup_choices)
    p_whatsup.add_argument("--root-dir", default="data/whatsup", help="Directory for WhatsUp metadata and image archives.")
    p_whatsup.add_argument("--metadata-file", default="", help="Local metadata JSON for one concrete WhatsUp dataset.")
    p_whatsup.add_argument("--image-root", default="", help="Optional directory containing extracted images.")
    p_whatsup.add_argument("--coco-root", default="", help="Optional COCO root reused by GSR-Bench-compatible WhatsUp helpers.")
    p_whatsup.add_argument("--gqa-root", default="", help="Optional Visual Genome/GQA image root reused by GSR-Bench-compatible helpers.")
    p_whatsup.add_argument("--output", required=True)
    p_whatsup.add_argument("--download", action="store_true", help="Download required metadata/images with gdown before processing.")
    p_whatsup.add_argument("--no-extract", action="store_true", help="Do not extract downloaded archives.")
    p_whatsup.add_argument("--metadata-only", action="store_true", help="Do not require local images; write image paths from metadata only.")
    p_whatsup.add_argument("--keep-non-left-right", action="store_true", help="Keep all parsed spatial relations for controls; default keeps only left/right.")
    p_whatsup.add_argument("--relations", default="", help="Comma-separated normalized relations to keep, for example above,below,inside,contains.")
    p_whatsup.add_argument("--relation-families", default="", help=f"Comma-separated relation families to keep. Choices: {','.join(sorted(RELATION_FAMILIES))}.")
    p_whatsup.add_argument("--positive-only", action="store_true", help="Keep only the first/correct caption option.")
    p_whatsup.add_argument("--limit", type=int, default=None)

    gsrbench_choices = sorted([*GSRBENCH_CONFIGS, *GSRBENCH_GROUPS])
    p_gsrbench = subparsers.add_parser(
        "prepare-gsrbench",
        help="Prepare the non-overlapping COCO/GQA two-object GSR-Bench extension.",
    )
    p_gsrbench.add_argument("--dataset", default="external_two_object", choices=gsrbench_choices)
    p_gsrbench.add_argument(
        "--root-dir",
        default="data/whatsup",
        help="Directory containing the official What'sUp/GSR-Bench metadata and image assets.",
    )
    p_gsrbench.add_argument("--metadata-file", default="", help="Local metadata JSON for one concrete split.")
    p_gsrbench.add_argument("--image-root", default="", help="Optional shared image directory for a single concrete split.")
    p_gsrbench.add_argument(
        "--coco-root",
        default="",
        help="Existing COCO 2017 root or val2017 directory; prevents downloading val2017.zip.",
    )
    p_gsrbench.add_argument(
        "--gqa-root",
        default="",
        help="Existing Visual Genome/GQA image root or vg_images directory.",
    )
    p_gsrbench.add_argument("--output", required=True)
    p_gsrbench.add_argument("--download", action="store_true", help="Download required official metadata/images with gdown.")
    p_gsrbench.add_argument("--no-extract", action="store_true")
    p_gsrbench.add_argument("--metadata-only", action="store_true")
    p_gsrbench.add_argument("--keep-non-left-right", action="store_true")
    p_gsrbench.add_argument("--relations", default="", help="Comma-separated normalized relations to keep.")
    p_gsrbench.add_argument(
        "--relation-families",
        default="",
        help=f"Comma-separated relation families to keep. Choices: {','.join(sorted(RELATION_FAMILIES))}.",
    )
    p_gsrbench.add_argument("--limit", type=int, default=None, help="Optional row limit per source split.")

    p_build = subparsers.add_parser("build-probes", help="Expand canonical samples into RECAP probes.")
    p_build.add_argument("--input", required=True)
    p_build.add_argument("--output", required=True)
    p_build.add_argument("--keep-non-left-right", action="store_true")
    p_build.add_argument("--probe-mode", default="full", choices=PROBE_MODES, help="Probe expansion set. profile_fast keeps the probes needed for the relation-profile detector.")

    p_validate_images = subparsers.add_parser("validate-images", help="Check whether non-text-only RECAP probes can load their images.")
    p_validate_images.add_argument("--input", required=True, help="Canonical sample JSONL or probe JSONL.")
    p_validate_images.add_argument("--output", default="", help="Optional JSON report path.")
    p_validate_images.add_argument("--is-probes", action="store_true", help="Input is already a probe JSONL; otherwise probes are built first.")
    p_validate_images.add_argument("--keep-non-left-right", action="store_true")
    p_validate_images.add_argument("--probe-mode", default="full", choices=PROBE_MODES, help="Probe expansion set used when --is-probes is not set.")
    p_validate_images.add_argument("--examples", type=int, default=20)

    p_audit = subparsers.add_parser("audit-data", help="Summarize canonical relation/task distributions before running probes.")
    p_audit.add_argument("--input", required=True)
    p_audit.add_argument("--output", required=True)
    p_audit.add_argument("--examples", type=int, default=3)

    p_prune_audit = subparsers.add_parser("prune-audit", help="Audit pruning task families, risk tags, and evidence-region coverage.")
    p_prune_audit.add_argument("--input", required=True)
    p_prune_audit.add_argument("--output", required=True)
    p_prune_audit.add_argument("--examples", type=int, default=3)
    p_prune_audit.add_argument("--limit", type=int, default=None)

    p_prune_offline = subparsers.add_parser("prune-offline-baselines", help="Build fixed-budget pruning mask baselines without loading a model.")
    p_prune_offline.add_argument("--input", required=True)
    p_prune_offline.add_argument("--output", required=True, help="JSONL output with one pruning mask record per sample/method/ratio.")
    p_prune_offline.add_argument("--summary-output", default="", help="Optional compact JSON summary path.")
    p_prune_offline.add_argument("--keep-ratios", default="0.15,0.25,0.35,0.5,0.7,1.0")
    p_prune_offline.add_argument("--selectors", default="random,grid,center")
    p_prune_offline.add_argument("--grid-size", default="24", help="Token grid as N or HxW. Default: 24.")
    p_prune_offline.add_argument("--seed", type=int, default=13)
    p_prune_offline.add_argument("--limit", type=int, default=None)

    p_attach_coco = subparsers.add_parser("attach-coco-evidence", help="Attach COCO instance boxes as evidence_regions to canonical samples.")
    p_attach_coco.add_argument("--input", required=True)
    p_attach_coco.add_argument("--output", required=True)
    p_attach_coco.add_argument("--instances-file", required=True, help="COCO instances_train/val JSON annotation file.")
    p_attach_coco.add_argument("--report-output", default="", help="Optional JSON report path.")
    p_attach_coco.add_argument("--min-area", type=float, default=1.0)

    p_ocr_regions = subparsers.add_parser("prepare-ocr-regions", help="Convert OCR/document JSON/JSONL with region boxes to canonical samples.")
    p_ocr_regions.add_argument("--input", required=True)
    p_ocr_regions.add_argument("--output", required=True)
    p_ocr_regions.add_argument("--image-root", default="")
    p_ocr_regions.add_argument("--dataset-name", default="OCR-Regions")
    p_ocr_regions.add_argument("--limit", type=int, default=None)

    p_download_textocr = subparsers.add_parser("download-textocr", help="Download official TextOCR annotation JSON.")
    p_download_textocr.add_argument("--split", default="val", choices=["train", "val", "validation"])
    p_download_textocr.add_argument("--output-dir", default="data/textocr")

    p_textocr_regions = subparsers.add_parser("prepare-textocr-regions", help="Convert TextOCR word boxes to canonical OCR evidence samples.")
    p_textocr_regions.add_argument("--annotation-file", default="", help="TextOCR_0.1_{split}.json. If omitted, download/use --output-dir.")
    p_textocr_regions.add_argument("--split", default="val", choices=["train", "val", "validation"])
    p_textocr_regions.add_argument("--output-dir", default="data/textocr")
    p_textocr_regions.add_argument("--image-root", default="")
    p_textocr_regions.add_argument("--output", required=True)
    p_textocr_regions.add_argument("--limit", type=int, default=None)
    p_textocr_regions.add_argument("--min-area", type=float, default=1.0)
    p_textocr_regions.add_argument("--max-regions", type=int, default=None)

    p_qwen = subparsers.add_parser("score-qwen-direct", help="Score probes with a direct Qwen2.5-VL likelihood backend.")
    p_qwen.add_argument("--probes", required=True)
    p_qwen.add_argument("--output", required=True)
    p_qwen.add_argument("--pretrained", required=True)
    p_qwen.add_argument("--device", default="cuda")
    p_qwen.add_argument("--device-map", default="auto")
    p_qwen.add_argument("--dtype", default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    p_qwen.add_argument("--attn-implementation", default="")
    p_qwen.add_argument("--min-pixels", type=int, default=50176)
    p_qwen.add_argument("--max-pixels", type=int, default=50176)
    p_qwen.add_argument("--use-fast-processor", action="store_true")
    p_qwen.add_argument("--debug-forward", action="store_true")
    p_qwen.add_argument("--target-delimiter", default=" ")
    p_qwen.add_argument("--allow-missing-images", action="store_true", help="Do not fail when a visual probe cannot load an image; useful only for debugging.")

    p_llava = subparsers.add_parser("score-llava-direct", help="Score probes with a direct LLaVA HF likelihood backend.")
    p_llava.add_argument("--probes", required=True)
    p_llava.add_argument("--output", required=True)
    p_llava.add_argument("--pretrained", required=True)
    p_llava.add_argument("--revision", default="main")
    p_llava.add_argument("--device", default="cuda")
    p_llava.add_argument("--device-map", default="auto")
    p_llava.add_argument("--dtype", default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    p_llava.add_argument("--trust-remote-code", action="store_true")
    p_llava.add_argument("--attn-implementation", default="")
    p_llava.add_argument("--chat-template", default="")
    p_llava.add_argument("--debug-forward", action="store_true")
    p_llava.add_argument("--target-delimiter", default=" ")
    p_llava.add_argument("--allow-missing-images", action="store_true", help="Do not fail when a visual probe cannot load an image; useful only for debugging.")

    p_internvl = subparsers.add_parser("score-internvl-direct", help="Score probes with a direct InternVL HF likelihood backend.")
    p_internvl.add_argument("--probes", required=True)
    p_internvl.add_argument("--output", required=True)
    p_internvl.add_argument("--pretrained", required=True)
    p_internvl.add_argument("--revision", default="main")
    p_internvl.add_argument("--device", default="cuda")
    p_internvl.add_argument("--device-map", default="auto")
    p_internvl.add_argument("--dtype", default="bfloat16", choices=["auto", "bfloat16", "float16", "float32"])
    p_internvl.add_argument("--trust-remote-code", action="store_true")
    p_internvl.add_argument("--low-cpu-mem-usage", action="store_true")
    p_internvl.add_argument("--attn-implementation", default="")
    p_internvl.add_argument("--min-patches", type=int, default=1)
    p_internvl.add_argument("--max-patches", type=int, default=12)
    p_internvl.add_argument("--debug-forward", action="store_true")
    p_internvl.add_argument("--target-delimiter", default=" ")
    p_internvl.add_argument("--allow-missing-images", action="store_true", help="Do not fail when a visual probe cannot load an image; useful only for debugging.")

    p_aggregate = subparsers.add_parser("aggregate", help="Aggregate yes/no probe losses into RECAP metrics.")
    p_aggregate.add_argument("--scores", required=True)
    p_aggregate.add_argument("--output", required=True)
    p_aggregate.add_argument("--samples-output", default="")

    p_recap_ablation = subparsers.add_parser("recap-ablation", help="Run offline RECAP component ablations from probe scores.")
    p_recap_ablation.add_argument("--scores", required=True, help="Probe scores JSONL from a RECAP run.")
    p_recap_ablation.add_argument("--output", required=True, help="Compact ablation JSON output path.")
    p_recap_ablation.add_argument("--coverage", type=int, default=80)
    p_recap_ablation.add_argument(
        "--risks",
        default="",
        help=f"Comma-separated ablation rows. Default: {','.join(DEFAULT_RECAP_ABLATION_RISKS)}",
    )
    p_recap_ablation.add_argument("--include-auprc", action="store_true", help="Also report AUPRC columns.")
    p_recap_ablation.add_argument("--include-by-family", dest="include_by_family", action="store_true", default=True)
    p_recap_ablation.add_argument("--no-by-family", dest="include_by_family", action="store_false")
    p_recap_ablation.add_argument("--include-by-relation", action="store_true", help="Also include per-relation rows.")

    p_recap_curves = subparsers.add_parser("recap-curves", help="Export risk-coverage curves from scored RECAP probes.")
    p_recap_curves.add_argument("--scores", required=True)
    p_recap_curves.add_argument("--output", required=True, help="JSON output path.")
    p_recap_curves.add_argument("--csv-output", default="", help="Optional flat CSV output path for plotting.")
    p_recap_curves.add_argument(
        "--risks",
        default="confidence,recap_evidence,rice_recap_selector,recap_img_pair,recap_anti_delta",
        help="Comma-separated risk names.",
    )
    p_recap_curves.add_argument("--min-coverage", type=int, default=10)
    p_recap_curves.add_argument("--max-coverage", type=int, default=100)
    p_recap_curves.add_argument("--step", type=int, default=5)

    p_recap_cost = subparsers.add_parser("recap-cost-utility", help="Export compute-utility metrics for low-cost RECAP variants.")
    p_recap_cost.add_argument("--scores", required=True)
    p_recap_cost.add_argument("--output", required=True)
    p_recap_cost.add_argument("--coverage", type=int, default=80)
    p_recap_cost.add_argument("--risks", default=",".join(LOW_COST_VARIANTS), help="Comma-separated risk names.")

    p_recap_bootstrap = subparsers.add_parser("recap-bootstrap-ci", help="Bootstrap confidence intervals from scored RECAP probes.")
    p_recap_bootstrap.add_argument("--scores", required=True)
    p_recap_bootstrap.add_argument("--output", required=True, help="JSON output path.")
    p_recap_bootstrap.add_argument("--csv-output", default="", help="Optional flat CSV output path.")
    p_recap_bootstrap.add_argument("--coverage", type=int, default=80)
    p_recap_bootstrap.add_argument("--risks", default=",".join(LOW_COST_VARIANTS), help="Comma-separated risk names.")
    p_recap_bootstrap.add_argument("--n-bootstrap", type=int, default=1000)
    p_recap_bootstrap.add_argument("--seed", type=int, default=13)
    p_recap_bootstrap.add_argument("--ci", type=float, default=0.95)

    p_low_cost_check = subparsers.add_parser("recap-low-cost-check", help="Verify low-cost RECAP variants from filtered probe subsets.")
    p_low_cost_check.add_argument("--scores", required=True)
    p_low_cost_check.add_argument("--output", required=True, help="JSON output path.")
    p_low_cost_check.add_argument("--csv-output", default="", help="Optional flat CSV output path.")
    p_low_cost_check.add_argument("--coverage", type=int, default=80)
    p_low_cost_check.add_argument("--variants", default=",".join(LOW_COST_VARIANTS), help="Comma-separated variant names.")
    p_low_cost_check.add_argument("--tolerance", type=float, default=1e-8)

    p_recap_cases = subparsers.add_parser("recap-case-studies", help="Extract qualitative RECAP case studies from scored probes.")
    p_recap_cases.add_argument("--scores", required=True)
    p_recap_cases.add_argument("--output", required=True, help="JSON output path.")
    p_recap_cases.add_argument("--jsonl-output", default="", help="Optional flattened JSONL case-study output path.")
    p_recap_cases.add_argument("--baseline", default="confidence")
    p_recap_cases.add_argument("--method", default="recap_evidence")
    p_recap_cases.add_argument("--coverage", type=int, default=80)
    p_recap_cases.add_argument("--examples", type=int, default=5)

    p_filter_recap = subparsers.add_parser("filter-recap-probes", help="Filter a full RECAP probe JSONL down to one low-cost variant.")
    p_filter_recap.add_argument("--input", required=True, help="Input probe JSONL.")
    p_filter_recap.add_argument("--output", required=True, help="Filtered probe JSONL.")
    p_filter_recap.add_argument(
        "--variant",
        required=True,
        choices=sorted(set(LOW_COST_VARIANTS) | {"selector_claim_delta", "selector_anti_delta", "selector_pair_delta", "selector_img_pair", "recap_text_pair"}),
    )

    p_compact = subparsers.add_parser("compact-metrics", help="Write a compact paper-table subset from a full metrics JSON.")
    p_compact.add_argument("--input", required=True)
    p_compact.add_argument("--output", required=True)
    p_compact.add_argument("--coverage", type=int, default=80)
    p_compact.add_argument("--risks", default="", help="Comma-separated risk prefixes. Default picks a main-table set.")
    p_compact.add_argument("--include-by-family", dest="include_by_family", action="store_true", default=True, help="Include main relation-family rows. This is the default.")
    p_compact.add_argument("--no-by-family", dest="include_by_family", action="store_false", help="Keep only overall rows.")
    p_compact.add_argument("--include-by-relation", action="store_true", help="Also include per-relation rows. Use only for debugging specific relations.")

    p_qwen_pilot = subparsers.add_parser("run-qwen-direct", help="Build probes, score with direct Qwen2.5-VL, and aggregate metrics.")
    p_qwen_pilot.add_argument("--input", required=True, help="Canonical RECAP JSONL.")
    p_qwen_pilot.add_argument("--work-dir", required=True)
    p_qwen_pilot.add_argument("--pretrained", required=True)
    p_qwen_pilot.add_argument("--is-probes", action="store_true", help="Input is already a yes/no probe JSONL; otherwise relation probes are built first.")
    p_qwen_pilot.add_argument("--device", default="cuda")
    p_qwen_pilot.add_argument("--device-map", default="auto")
    p_qwen_pilot.add_argument("--dtype", default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    p_qwen_pilot.add_argument("--attn-implementation", default="")
    p_qwen_pilot.add_argument("--min-pixels", type=int, default=50176)
    p_qwen_pilot.add_argument("--max-pixels", type=int, default=50176)
    p_qwen_pilot.add_argument("--use-fast-processor", action="store_true")
    p_qwen_pilot.add_argument("--debug-forward", action="store_true")
    p_qwen_pilot.add_argument("--target-delimiter", default=" ")
    p_qwen_pilot.add_argument("--allow-missing-images", action="store_true", help="Do not fail when a visual probe cannot load an image; useful only for debugging.")
    p_qwen_pilot.add_argument("--keep-non-left-right", action="store_true", help="Build probes for all relations in the input instead of filtering to left/right.")
    p_qwen_pilot.add_argument("--probe-mode", default="full", choices=PROBE_MODES, help="Probe expansion set. Defaults to full for exact old behavior.")
    p_qwen_pilot.add_argument("--limit", type=int, default=None, help="Optional sample limit for model smoke tests.")

    p_qwen_pruned = subparsers.add_parser("run-qwen-pruned", help="Score Qwen-VL after pruning visual placeholder tokens before LLM prefill.")
    p_qwen_pruned.add_argument("--input", required=True, help="Canonical RECAP JSONL.")
    p_qwen_pruned.add_argument("--work-dir", required=True)
    p_qwen_pruned.add_argument("--pretrained", required=True)
    p_qwen_pruned.add_argument("--is-probes", action="store_true", help="Input is already a yes/no probe JSONL; otherwise relation probes are built first.")
    p_qwen_pruned.add_argument("--selector", default="rise", help="Visual token selector: random, grid, grid_topk, center, topk, shuffled_topk, rise, hybrid, rel_hybrid, soft_relboost, protected_rel_hybrid, protected_topk, protected_center_topk, soft_evidence_topk, Qwen3 visionzip, or non-oracle embed_topk/embed_hybrid/embed_rise/embed_grid_topk/target_embed_topk/target_embed_grid_topk/target_embed_protected_topk/target_embed_protected_center_topk/target_embed_soft_evidence_topk.")
    p_qwen_pruned.add_argument("--keep-ratio", type=float, default=0.35)
    p_qwen_pruned.add_argument("--budget-mode", default="fixed", choices=["fixed", "risk_adaptive", "risk_bucket", "sensitivity_policy", "evidence_saturation"])
    p_qwen_pruned.add_argument("--risk-scores", default="", help="Sample score JSONL used by risk_adaptive budgets.")
    p_qwen_pruned.add_argument("--risk-key", default="recap_evidence_risk")
    p_qwen_pruned.add_argument("--budget-ratios", default="", help="Sample-level keep-ratio JSONL used by sensitivity_policy budgets.")
    p_qwen_pruned.add_argument("--budget-ratio-key", default="keep_ratio", help="Field name in --budget-ratios containing the keep ratio.")
    p_qwen_pruned.add_argument("--kept-indices", default="", help="Sample-level visual token indices JSONL. Overrides selector output when provided.")
    p_qwen_pruned.add_argument("--kept-indices-key", default="kept_indices", help="Field name in --kept-indices containing a list of indices.")
    p_qwen_pruned.add_argument("--rho-min", type=float, default=0.15)
    p_qwen_pruned.add_argument("--rho-max", type=float, default=0.70)
    p_qwen_pruned.add_argument("--hybrid-core-ratio", type=float, default=0.50, help="Hybrid selector budget share for evidence top tokens.")
    p_qwen_pruned.add_argument("--hybrid-context-ratio", type=float, default=0.25, help="Hybrid selector budget share for evidence-neighborhood context tokens.")
    p_qwen_pruned.add_argument("--evidence-boost", type=float, default=0.10, help="Soft evidence score boost used by soft_evidence_topk selectors.")
    p_qwen_pruned.add_argument("--embedding-relevance-weight", type=float, default=0.85, help="Weight on target relevance; one minus this value weights visual-token norm.")
    p_qwen_pruned.add_argument("--embedding-query-topk", type=int, default=2, help="Number of strongest query-token similarities averaged per visual token.")
    p_qwen_pruned.add_argument("--saturation-temperature", type=float, default=0.12)
    p_qwen_pruned.add_argument("--saturation-mass-target", type=float, default=0.72)
    p_qwen_pruned.add_argument("--saturation-cell-target", type=float, default=0.75)
    p_qwen_pruned.add_argument("--seed", type=int, default=13)
    p_qwen_pruned.add_argument("--device", default="cuda")
    p_qwen_pruned.add_argument("--device-map", default="auto")
    p_qwen_pruned.add_argument("--dtype", default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    p_qwen_pruned.add_argument("--attn-implementation", default="")
    p_qwen_pruned.add_argument("--min-pixels", type=int, default=50176)
    p_qwen_pruned.add_argument("--max-pixels", type=int, default=50176)
    p_qwen_pruned.add_argument("--use-fast-processor", action="store_true")
    p_qwen_pruned.add_argument("--debug-forward", action="store_true")
    p_qwen_pruned.add_argument("--target-delimiter", default=" ")
    p_qwen_pruned.add_argument("--allow-missing-images", action="store_true", help="Do not fail when a visual probe cannot load an image; useful only for debugging.")
    p_qwen_pruned.add_argument("--keep-non-left-right", action="store_true", help="Build probes for all relations in the input instead of filtering to left/right.")
    p_qwen_pruned.add_argument("--probe-mode", default="profile_fast", choices=PROBE_MODES, help="Probe expansion set before optional --orig-only filtering.")
    p_qwen_pruned.add_argument("--orig-only", action="store_true", help="Only score orig probes for pruning accuracy/latency curves.")
    p_qwen_pruned.add_argument("--limit", type=int, default=None, help="Optional sample limit for model smoke tests.")

    p_llava_pruned = subparsers.add_parser("run-llava-pruned", help="Score LLaVA after pruning visual placeholder tokens before LLM prefill.")
    p_llava_pruned.add_argument("--input", required=True, help="Canonical RECAP JSONL.")
    p_llava_pruned.add_argument("--work-dir", required=True)
    p_llava_pruned.add_argument("--pretrained", required=True)
    p_llava_pruned.add_argument("--is-probes", action="store_true", help="Input is already a yes/no probe JSONL.")
    p_llava_pruned.add_argument("--selector", default="target_embed_topk", help="Visual token selector, including non-oracle embed_topk/target_embed_topk/target_embed_grid_topk variants and LLaVA-only external-method ports: visionzip, fastv, scope, coin, anchorprune.")
    p_llava_pruned.add_argument("--keep-ratio", type=float, default=0.30)
    p_llava_pruned.add_argument("--hybrid-core-ratio", type=float, default=0.50)
    p_llava_pruned.add_argument("--hybrid-context-ratio", type=float, default=0.25)
    p_llava_pruned.add_argument("--evidence-boost", type=float, default=0.10)
    p_llava_pruned.add_argument("--scope-alpha", type=float, default=1.0, help="SCOPE saliency exponent; alpha=0 is the pure feature-coverage setting.")
    p_llava_pruned.add_argument("--coin-alpha", type=float, default=0.90, help="CoIn coverage weight (paper setting for LLaVA-1.5 at 128 tokens: 0.90).")
    p_llava_pruned.add_argument("--coin-beta", type=float, default=0.60, help="CoIn visual-saliency weight (paper setting for LLaVA-1.5 at 128 tokens: 0.60).")
    p_llava_pruned.add_argument("--anchorprune-k-min", type=int, default=0, help="Minimum protected anchor size; zero uses --anchorprune-k-min-ratio.")
    p_llava_pruned.add_argument("--anchorprune-k-min-ratio", type=float, default=0.15625, help="Ratio-derived K_min used for unmatched budgets (official settings use 5/32, 10/64, and 20/128).")
    p_llava_pruned.add_argument("--anchorprune-tau", type=float, default=0.20)
    p_llava_pruned.add_argument("--anchorprune-patience", type=int, default=3)
    p_llava_pruned.add_argument("--anchorprune-kmax-ratio", type=float, default=0.50)
    p_llava_pruned.add_argument("--anchorprune-clip-model", default="openai/clip-vit-large-patch14-336")
    p_llava_pruned.add_argument("--kept-indices", default="", help="Sample-level visual token indices JSONL. Overrides selector output when provided.")
    p_llava_pruned.add_argument("--kept-indices-key", default="kept_indices", help="Field name in --kept-indices containing a list of indices.")
    p_llava_pruned.add_argument(
        "--position-mode",
        default="compact",
        choices=["compact", "preserve"],
        help="Logical position IDs after physical token deletion: renumber contiguously or preserve pre-pruning positions.",
    )
    p_llava_pruned.add_argument("--seed", type=int, default=13)
    p_llava_pruned.add_argument("--revision", default="main")
    p_llava_pruned.add_argument("--device", default="cuda")
    p_llava_pruned.add_argument("--device-map", default="auto")
    p_llava_pruned.add_argument("--dtype", default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    p_llava_pruned.add_argument("--trust-remote-code", action="store_true")
    p_llava_pruned.add_argument("--attn-implementation", default="")
    p_llava_pruned.add_argument("--chat-template", default="")
    p_llava_pruned.add_argument("--debug-forward", action="store_true")
    p_llava_pruned.add_argument("--target-delimiter", default=" ")
    p_llava_pruned.add_argument("--allow-missing-images", action="store_true", help="Do not fail when a visual probe cannot load an image; useful only for debugging.")
    p_llava_pruned.add_argument("--keep-non-left-right", action="store_true", help="Build probes for all relations in the input instead of filtering to left/right.")
    p_llava_pruned.add_argument("--probe-mode", default="profile_fast", choices=PROBE_MODES, help="Probe expansion set before optional --orig-only filtering.")
    p_llava_pruned.add_argument("--orig-only", action="store_true", help="Only score orig probes for pruning accuracy/latency curves.")
    p_llava_pruned.add_argument("--limit", type=int, default=None, help="Optional sample limit for model smoke tests.")

    p_internvl_pruned = subparsers.add_parser("run-internvl-pruned", help="Score InternVL after pruning visual placeholder tokens before LLM prefill.")
    p_internvl_pruned.add_argument("--input", required=True, help="Canonical RECAP JSONL.")
    p_internvl_pruned.add_argument("--work-dir", required=True)
    p_internvl_pruned.add_argument("--pretrained", required=True)
    p_internvl_pruned.add_argument("--is-probes", action="store_true", help="Input is already a yes/no probe JSONL.")
    p_internvl_pruned.add_argument("--selector", default="target_embed_topk", help="Visual token selector, including non-oracle embed_topk/target_embed_topk/target_embed_grid_topk variants.")
    p_internvl_pruned.add_argument("--keep-ratio", type=float, default=0.50)
    p_internvl_pruned.add_argument("--hybrid-core-ratio", type=float, default=0.50)
    p_internvl_pruned.add_argument("--hybrid-context-ratio", type=float, default=0.25)
    p_internvl_pruned.add_argument("--evidence-boost", type=float, default=0.10)
    p_internvl_pruned.add_argument("--seed", type=int, default=13)
    p_internvl_pruned.add_argument("--kept-indices", default="", help="Sample-level visual token indices JSONL. Overrides selector output when provided.")
    p_internvl_pruned.add_argument("--kept-indices-key", default="kept_indices", help="Field name in --kept-indices containing a list of indices.")
    p_internvl_pruned.add_argument(
        "--position-mode",
        default="compact",
        choices=["compact", "preserve"],
        help="Logical position IDs after physical token deletion: renumber contiguously or preserve pre-pruning positions.",
    )
    p_internvl_pruned.add_argument("--revision", default="main")
    p_internvl_pruned.add_argument("--device", default="cuda")
    p_internvl_pruned.add_argument("--device-map", default="auto")
    p_internvl_pruned.add_argument("--dtype", default="bfloat16", choices=["auto", "bfloat16", "float16", "float32"])
    p_internvl_pruned.add_argument("--trust-remote-code", action="store_true")
    p_internvl_pruned.add_argument("--low-cpu-mem-usage", action="store_true")
    p_internvl_pruned.add_argument("--attn-implementation", default="")
    p_internvl_pruned.add_argument("--min-patches", type=int, default=1)
    p_internvl_pruned.add_argument("--max-patches", type=int, default=12)
    p_internvl_pruned.add_argument("--debug-forward", action="store_true")
    p_internvl_pruned.add_argument("--target-delimiter", default=" ")
    p_internvl_pruned.add_argument("--allow-missing-images", action="store_true", help="Do not fail when a visual probe cannot load an image; useful only for debugging.")
    p_internvl_pruned.add_argument("--keep-non-left-right", action="store_true", help="Build probes for all relations in the input instead of filtering to left/right.")
    p_internvl_pruned.add_argument("--probe-mode", default="profile_fast", choices=PROBE_MODES, help="Probe expansion set before optional --orig-only filtering.")
    p_internvl_pruned.add_argument("--orig-only", action="store_true", help="Only score orig probes for pruning accuracy/latency curves.")
    p_internvl_pruned.add_argument("--limit", type=int, default=None, help="Optional sample limit for model smoke tests.")

    p_llava_pilot = subparsers.add_parser("run-llava-direct", help="Build probes, score with direct LLaVA HF, and aggregate metrics.")
    p_llava_pilot.add_argument("--input", required=True, help="Canonical RECAP JSONL.")
    p_llava_pilot.add_argument("--work-dir", required=True)
    p_llava_pilot.add_argument("--pretrained", required=True)
    p_llava_pilot.add_argument("--is-probes", action="store_true", help="Input is already a yes/no probe JSONL.")
    p_llava_pilot.add_argument("--revision", default="main")
    p_llava_pilot.add_argument("--device", default="cuda")
    p_llava_pilot.add_argument("--device-map", default="auto")
    p_llava_pilot.add_argument("--dtype", default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    p_llava_pilot.add_argument("--trust-remote-code", action="store_true")
    p_llava_pilot.add_argument("--attn-implementation", default="")
    p_llava_pilot.add_argument("--chat-template", default="")
    p_llava_pilot.add_argument("--debug-forward", action="store_true")
    p_llava_pilot.add_argument("--target-delimiter", default=" ")
    p_llava_pilot.add_argument("--allow-missing-images", action="store_true", help="Do not fail when a visual probe cannot load an image; useful only for debugging.")
    p_llava_pilot.add_argument("--keep-non-left-right", action="store_true", help="Build probes for all relations in the input instead of filtering to left/right.")
    p_llava_pilot.add_argument("--probe-mode", default="full", choices=PROBE_MODES, help="Probe expansion set. profile_fast is useful for slow HF backends.")
    p_llava_pilot.add_argument("--orig-only", action="store_true", help="Only score orig probes for main accuracy/latency tables.")
    p_llava_pilot.add_argument("--limit", type=int, default=None, help="Optional sample limit for model smoke tests.")

    p_internvl_pilot = subparsers.add_parser("run-internvl-direct", help="Build probes, score with direct InternVL HF, and aggregate metrics.")
    p_internvl_pilot.add_argument("--input", required=True, help="Canonical RECAP JSONL.")
    p_internvl_pilot.add_argument("--work-dir", required=True)
    p_internvl_pilot.add_argument("--pretrained", required=True)
    p_internvl_pilot.add_argument("--is-probes", action="store_true", help="Input is already a yes/no probe JSONL.")
    p_internvl_pilot.add_argument("--revision", default="main")
    p_internvl_pilot.add_argument("--device", default="cuda")
    p_internvl_pilot.add_argument("--device-map", default="auto")
    p_internvl_pilot.add_argument("--dtype", default="bfloat16", choices=["auto", "bfloat16", "float16", "float32"])
    p_internvl_pilot.add_argument("--trust-remote-code", action="store_true")
    p_internvl_pilot.add_argument("--low-cpu-mem-usage", action="store_true")
    p_internvl_pilot.add_argument("--attn-implementation", default="")
    p_internvl_pilot.add_argument("--min-patches", type=int, default=1)
    p_internvl_pilot.add_argument("--max-patches", type=int, default=12)
    p_internvl_pilot.add_argument("--debug-forward", action="store_true")
    p_internvl_pilot.add_argument("--target-delimiter", default=" ")
    p_internvl_pilot.add_argument("--allow-missing-images", action="store_true", help="Do not fail when a visual probe cannot load an image; useful only for debugging.")
    p_internvl_pilot.add_argument("--keep-non-left-right", action="store_true", help="Build probes for all relations in the input instead of filtering to left/right.")
    p_internvl_pilot.add_argument("--probe-mode", default="full", choices=PROBE_MODES, help="Probe expansion set. profile_fast is useful for slow HF backends.")
    p_internvl_pilot.add_argument("--orig-only", action="store_true", help="Only score orig probes for main accuracy/latency tables.")
    p_internvl_pilot.add_argument("--limit", type=int, default=None, help="Optional sample limit for model smoke tests.")

    args = parser.parse_args()

    if args.command == "prepare-vsr":
        canonical = prepare_vsr(
            split=args.split,
            dataset_path=args.dataset_path,
            metadata_file=args.metadata_file or None,
            coco_root=args.coco_root or None,
            output_image_dir=args.image_dir or None,
            download_images=not args.metadata_only,
            local_files_only=args.local_files_only,
            cache_dir=args.cache_dir,
            hf_endpoint=args.hf_endpoint or None,
            limit=args.limit,
            left_right_only=not args.keep_non_left_right and not args.relations and not args.relation_families,
            relations=_parse_relations(args.relations),
            relation_families=_parse_relation_families(args.relation_families),
            allow_missing_images=args.allow_missing_images or args.metadata_only,
        )
        write_jsonl(args.output, canonical)
        print(f"Wrote {len(canonical)} VSR canonical samples to {args.output}")
        return

    if args.command == "prepare-whatsup":
        canonical = prepare_whatsup(
            dataset=args.dataset,
            root_dir=args.root_dir,
            metadata_file=args.metadata_file or None,
            image_root=args.image_root or None,
            download=args.download,
            extract=not args.no_extract,
            metadata_only=args.metadata_only,
            left_right_only=not args.keep_non_left_right and not args.relations and not args.relation_families,
            relations=_parse_relations(args.relations),
            relation_families=_parse_relation_families(args.relation_families),
            positive_only=args.positive_only,
            limit=args.limit,
        )
        write_jsonl(args.output, canonical)
        print(f"Wrote {len(canonical)} WhatsUp canonical samples to {args.output}")
        return

    if args.command == "prepare-gsrbench":
        canonical = prepare_gsrbench(
            dataset=args.dataset,
            root_dir=args.root_dir,
            metadata_file=args.metadata_file or None,
            image_root=args.image_root or None,
            coco_root=args.coco_root or None,
            gqa_root=args.gqa_root or None,
            download=args.download,
            extract=not args.no_extract,
            metadata_only=args.metadata_only,
            left_right_only=not args.keep_non_left_right and not args.relations and not args.relation_families,
            relations=_parse_relations(args.relations),
            relation_families=_parse_relation_families(args.relation_families),
            limit=args.limit,
        )
        write_jsonl(args.output, canonical)
        print(f"Wrote {len(canonical)} GSR-Bench canonical samples to {args.output}")
        return

    if args.command == "build-probes":
        samples = read_jsonl(args.input)
        probes = samples if args.is_probes else build_probe_dataset(samples, require_left_right=not args.keep_non_left_right, probe_mode=args.probe_mode)
        write_jsonl(args.output, probes)
        print(f"Wrote {len(probes)} probes from {len(samples)} samples to {args.output}")
        return

    if args.command == "validate-images":
        rows = read_jsonl(args.input)
        probes = rows if args.is_probes else build_probe_dataset(rows, require_left_right=not args.keep_non_left_right, probe_mode=args.probe_mode)
        report = validate_probe_images(probes, limit_examples=args.examples)
        if args.output:
            write_json(args.output, report)
        print(report)
        if report["missing_visual_count"]:
            raise SystemExit(2)
        return

    if args.command == "audit-data":
        samples = read_jsonl(args.input)
        report = audit_samples(samples, examples_per_relation=args.examples)
        write_json(args.output, report)
        print(
            f"Audit rows={report['num_rows']} relations={report['relations']} "
            f"answers={report['answers']} choice_groups={report['choice_groups']['num_groups']}"
        )
        return

    if args.command == "prune-audit":
        from recap.prune.audit import audit_prune_samples

        samples = read_jsonl(args.input)
        if args.limit is not None:
            samples = samples[: args.limit]
        report = audit_prune_samples(samples, examples_per_family=args.examples)
        write_json(args.output, report)
        print(
            f"Prune audit rows={report['num_samples']} "
            f"families={report['task_families']} evidence={report['evidence']}"
        )
        return

    if args.command == "prune-offline-baselines":
        from recap.prune.budgets import parse_keep_ratios
        from recap.prune.offline import build_offline_prune_baselines, summarize_offline_prune_records
        from recap.prune.selectors import parse_selectors

        samples = read_jsonl(args.input)
        if args.limit is not None:
            samples = samples[: args.limit]
        grid_rows, grid_cols = _parse_grid_shape(args.grid_size)
        records = build_offline_prune_baselines(
            samples,
            keep_ratios=parse_keep_ratios(args.keep_ratios),
            selectors=parse_selectors(args.selectors),
            grid_rows=grid_rows,
            grid_cols=grid_cols,
            seed=args.seed,
        )
        write_jsonl(args.output, records)
        summary = summarize_offline_prune_records(records)
        if args.summary_output:
            write_json(args.summary_output, summary)
        print(
            f"Wrote {len(records)} offline pruning records from {len(samples)} samples "
            f"to {args.output}"
        )
        return

    if args.command == "attach-coco-evidence":
        from recap.coco_evidence import attach_coco_evidence

        samples = read_jsonl(args.input)
        enriched, report = attach_coco_evidence(
            samples,
            instances_file=args.instances_file,
            min_area=args.min_area,
        )
        write_jsonl(args.output, enriched)
        if args.report_output:
            write_json(args.report_output, report)
        print(
            f"Attached COCO evidence to {report['matched_any']}/{report['num_samples']} samples "
            f"(both={report['matched_both']}) and wrote {args.output}"
        )
        return

    if args.command == "prepare-ocr-regions":
        from recap.ocr_regions import prepare_ocr_region_samples

        samples = prepare_ocr_region_samples(
            input_path=args.input,
            image_root=args.image_root or None,
            dataset_name=args.dataset_name,
            limit=args.limit,
        )
        write_jsonl(args.output, samples)
        with_evidence = sum(1 for sample in samples if sample.get("evidence_regions"))
        print(f"Wrote {len(samples)} OCR-region samples to {args.output}; with_evidence={with_evidence}")
        return

    if args.command == "download-textocr":
        from recap.textocr_regions import download_textocr_annotations

        path = download_textocr_annotations(split=args.split, output_dir=args.output_dir)
        print(f"TextOCR annotation ready: {path}")
        return

    if args.command == "prepare-textocr-regions":
        from recap.textocr_regions import download_textocr_annotations, prepare_textocr_region_samples

        annotation_file = args.annotation_file or download_textocr_annotations(
            split=args.split,
            output_dir=args.output_dir,
        )
        samples = prepare_textocr_region_samples(
            annotation_file=annotation_file,
            image_root=args.image_root or None,
            limit=args.limit,
            min_area=args.min_area,
            max_regions=args.max_regions,
        )
        write_jsonl(args.output, samples)
        print(f"Wrote {len(samples)} TextOCR region samples to {args.output}")
        return

    if args.command == "score-qwen-direct":
        from recap.qwen_direct_backend import score_probes_with_qwen_direct

        probes = read_jsonl(args.probes)
        scores = score_probes_with_qwen_direct(
            probes,
            pretrained=args.pretrained,
            device=args.device,
            device_map=args.device_map,
            dtype=args.dtype,
            attn_implementation=args.attn_implementation or None,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
            use_fast_processor=args.use_fast_processor,
            target_delimiter=args.target_delimiter,
            debug_forward=args.debug_forward,
            strict_images=not args.allow_missing_images,
        )
        write_jsonl(args.output, scores)
        print(f"Wrote {len(scores)} Qwen direct scored probes to {args.output}")
        return

    if args.command == "score-llava-direct":
        from recap.llava_direct_backend import score_probes_with_llava_direct

        probes = read_jsonl(args.probes)
        scores = score_probes_with_llava_direct(
            probes,
            pretrained=args.pretrained,
            revision=args.revision,
            device=args.device,
            device_map=args.device_map,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
            attn_implementation=args.attn_implementation or None,
            chat_template=args.chat_template or None,
            target_delimiter=args.target_delimiter,
            debug_forward=args.debug_forward,
            strict_images=not args.allow_missing_images,
        )
        write_jsonl(args.output, scores)
        print(f"Wrote {len(scores)} LLaVA direct scored probes to {args.output}")
        return

    if args.command == "score-internvl-direct":
        from recap.internvl_direct_backend import score_probes_with_internvl_direct

        probes = read_jsonl(args.probes)
        scores = score_probes_with_internvl_direct(
            probes,
            pretrained=args.pretrained,
            revision=args.revision,
            device=args.device,
            device_map=args.device_map,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
            low_cpu_mem_usage=args.low_cpu_mem_usage,
            attn_implementation=args.attn_implementation or None,
            min_patches=args.min_patches,
            max_patches=args.max_patches,
            target_delimiter=args.target_delimiter,
            debug_forward=args.debug_forward,
            strict_images=not args.allow_missing_images,
        )
        write_jsonl(args.output, scores)
        print(f"Wrote {len(scores)} InternVL direct scored probes to {args.output}")
        return

    if args.command == "aggregate":
        scores = read_jsonl(args.scores)
        result = aggregate_scores(scores)
        write_json(args.output, result["metrics"])
        if args.samples_output:
            write_jsonl(args.samples_output, result["samples"])
        print(f"Wrote metrics for {len(result['samples'])} samples to {args.output}")
        return

    if args.command == "recap-ablation":
        scores = read_jsonl(args.scores)
        risks = [item.strip() for item in args.risks.split(",") if item.strip()] if args.risks else None
        report = ablate_recap_probe_scores(
            scores,
            coverage=args.coverage,
            risks=risks,
            include_by_family=args.include_by_family,
            include_by_relation=args.include_by_relation,
            include_auprc=args.include_auprc,
        )
        write_json(args.output, report)
        print(f"Wrote RECAP ablation report for {int(report['overview']['num_samples'])} samples to {args.output}")
        return

    if args.command == "recap-curves":
        scores = read_jsonl(args.scores)
        risks = [item.strip() for item in args.risks.split(",") if item.strip()] if args.risks else None
        report = recap_coverage_curves(
            scores,
            risks=risks,
            min_coverage=args.min_coverage,
            max_coverage=args.max_coverage,
            step=args.step,
        )
        write_json(args.output, report)
        if args.csv_output:
            write_csv(args.csv_output, coverage_curves_to_csv_rows(report))
        print(f"Wrote RECAP coverage curves for {int(report['overview']['num_samples'])} samples to {args.output}")
        return

    if args.command == "recap-cost-utility":
        scores = read_jsonl(args.scores)
        risks = [item.strip() for item in args.risks.split(",") if item.strip()] if args.risks else None
        report = recap_cost_utility(scores, risks=risks, coverage=args.coverage)
        write_json(args.output, report)
        print(f"Wrote RECAP cost-utility report for {int(report['overview']['num_samples'])} samples to {args.output}")
        return

    if args.command == "recap-bootstrap-ci":
        scores = read_jsonl(args.scores)
        risks = [item.strip() for item in args.risks.split(",") if item.strip()] if args.risks else None
        report = recap_bootstrap_ci(
            scores,
            risks=risks,
            coverage=args.coverage,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
            ci=args.ci,
        )
        write_json(args.output, report)
        if args.csv_output:
            write_csv(args.csv_output, bootstrap_ci_to_csv_rows(report))
        print(f"Wrote RECAP bootstrap CI report for {int(report['overview']['num_samples'])} samples to {args.output}")
        return

    if args.command == "recap-low-cost-check":
        scores = read_jsonl(args.scores)
        variants = [item.strip() for item in args.variants.split(",") if item.strip()] if args.variants else None
        report = recap_low_cost_equivalence(
            scores,
            variants=variants,
            coverage=args.coverage,
            tolerance=args.tolerance,
        )
        write_json(args.output, report)
        if args.csv_output:
            write_csv(args.csv_output, low_cost_equivalence_to_csv_rows(report))
        print(f"Wrote RECAP low-cost equivalence report for {int(report['overview']['num_samples'])} samples to {args.output}")
        return

    if args.command == "recap-case-studies":
        scores = read_jsonl(args.scores)
        report = extract_recap_case_studies(
            scores,
            baseline=args.baseline,
            method=args.method,
            coverage=args.coverage,
            examples=args.examples,
        )
        write_json(args.output, report)
        if args.jsonl_output:
            write_case_jsonl(args.jsonl_output, report)
        print(f"Wrote RECAP case studies to {args.output}")
        return

    if args.command == "filter-recap-probes":
        probes = read_jsonl(args.input)
        filtered = filter_recap_probes(probes, variant=args.variant)
        write_jsonl(args.output, filtered)
        print(f"Wrote {len(filtered)} filtered probes from {len(probes)} input probes to {args.output}")
        return

    if args.command == "compact-metrics":
        metrics = read_json(args.input)
        risks = [item.strip() for item in args.risks.split(",") if item.strip()] if args.risks else None
        write_json(
            args.output,
            compact_metrics(
                metrics,
                risks=risks,
                coverage=args.coverage,
                include_by_family=args.include_by_family,
                include_by_relation=args.include_by_relation,
            ),
        )
        print(f"Wrote compact metrics to {args.output}")
        return

    if args.command == "run-qwen-direct":
        from recap.qwen_direct_backend import score_probes_with_qwen_direct

        work_dir = Path(args.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        samples = read_jsonl(args.input)
        if args.limit is not None:
            samples = samples[: args.limit]
        probes = samples if args.is_probes else build_probe_dataset(samples, require_left_right=not args.keep_non_left_right, probe_mode=args.probe_mode)
        probes_path = work_dir / "probes.jsonl"
        scores_path = work_dir / "probe_scores.jsonl"
        metrics_path = work_dir / "metrics.json"
        samples_path = work_dir / "sample_scores.jsonl"
        write_jsonl(probes_path, probes)
        scores = score_probes_with_qwen_direct(
            probes,
            pretrained=args.pretrained,
            device=args.device,
            device_map=args.device_map,
            dtype=args.dtype,
            attn_implementation=args.attn_implementation or None,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
            use_fast_processor=args.use_fast_processor,
            target_delimiter=args.target_delimiter,
            debug_forward=args.debug_forward,
            strict_images=not args.allow_missing_images,
        )
        write_jsonl(scores_path, scores)
        result = aggregate_scores(scores)
        write_json(metrics_path, result["metrics"])
        write_jsonl(samples_path, result["samples"])
        print(f"Wrote {len(probes)} probes, {len(scores)} Qwen direct probe scores, and metrics to {work_dir}")
        return

    if args.command == "run-qwen-pruned":
        from recap.qwen_pruned_backend import PruneConfig, score_probes_with_qwen_pruned, summarize_prune_traces

        work_dir = Path(args.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        samples = read_jsonl(args.input)
        if args.limit is not None:
            samples = samples[: args.limit]
        probes = samples if args.is_probes else build_probe_dataset(samples, require_left_right=not args.keep_non_left_right, probe_mode=args.probe_mode)
        if args.orig_only:
            probes = [probe for probe in probes if probe.get("probe") == "orig"]
            for probe in probes:
                probe["probe_count"] = 1
        probes_path = work_dir / "probes.jsonl"
        scores_path = work_dir / "probe_scores.jsonl"
        traces_path = work_dir / "prune_traces.jsonl"
        metrics_path = work_dir / "metrics.json"
        samples_path = work_dir / "sample_scores.jsonl"
        write_jsonl(probes_path, probes)

        risk_scores = None
        if args.risk_scores:
            risk_scores = _load_ranked_risk_scores(args.risk_scores, args.risk_key)
        elif args.budget_mode in {"risk_adaptive", "risk_bucket"}:
            raise ValueError(f"--budget-mode {args.budget_mode} requires --risk-scores.")
        budget_ratios = None
        if args.budget_ratios:
            budget_ratios = _load_budget_ratios(args.budget_ratios, args.budget_ratio_key)
        elif args.budget_mode == "sensitivity_policy":
            raise ValueError("--budget-mode sensitivity_policy requires --budget-ratios.")
        kept_indices_by_sample = None
        if args.kept_indices:
            kept_indices_by_sample = _load_kept_indices(args.kept_indices, args.kept_indices_key)

        prune_config = PruneConfig(
            selector=args.selector,
            keep_ratio=args.keep_ratio,
            budget_mode=args.budget_mode,
            rho_min=args.rho_min,
            rho_max=args.rho_max,
            seed=args.seed,
            hybrid_core_ratio=args.hybrid_core_ratio,
            hybrid_context_ratio=args.hybrid_context_ratio,
            evidence_boost=args.evidence_boost,
            embedding_relevance_weight=args.embedding_relevance_weight,
            embedding_query_topk=args.embedding_query_topk,
            saturation_temperature=args.saturation_temperature,
            saturation_mass_target=args.saturation_mass_target,
            saturation_cell_target=args.saturation_cell_target,
        )
        scores, traces = score_probes_with_qwen_pruned(
            probes,
            pretrained=args.pretrained,
            prune_config=prune_config,
            risk_scores=risk_scores,
            budget_ratios=budget_ratios,
            kept_indices_by_sample=kept_indices_by_sample,
            device=args.device,
            device_map=args.device_map,
            dtype=args.dtype,
            attn_implementation=args.attn_implementation or None,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
            use_fast_processor=args.use_fast_processor,
            target_delimiter=args.target_delimiter,
            debug_forward=args.debug_forward,
            strict_images=not args.allow_missing_images,
        )
        write_jsonl(scores_path, scores)
        write_jsonl(traces_path, traces)
        result = aggregate_scores(scores)
        metrics = result["metrics"]
        metrics["pruning"] = summarize_prune_traces(traces)
        metrics["prune_selector"] = args.selector
        metrics["prune_score_source"] = traces[0].get("score_source", "") if traces else ""
        metrics["prune_budget_mode"] = args.budget_mode
        metrics["prune_target_keep_ratio"] = float(args.keep_ratio)
        metrics["prune_risk_key"] = args.risk_key if args.risk_scores else ""
        metrics["prune_budget_ratio_key"] = args.budget_ratio_key if args.budget_ratios else ""
        metrics["prune_kept_indices_key"] = args.kept_indices_key if args.kept_indices else ""
        write_json(metrics_path, metrics)
        write_jsonl(samples_path, result["samples"])
        print(
            f"Wrote {len(probes)} probes, {len(scores)} Qwen pruned probe scores, "
            f"{len(traces)} traces, and metrics to {work_dir}"
        )
        return

    if args.command == "run-llava-pruned":
        from recap.llava_pruned_backend import LlavaPruneConfig, score_probes_with_llava_pruned, summarize_llava_prune_traces

        work_dir = Path(args.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        samples = read_jsonl(args.input)
        if args.limit is not None:
            samples = samples[: args.limit]
        probes = samples if args.is_probes else build_probe_dataset(samples, require_left_right=not args.keep_non_left_right, probe_mode=args.probe_mode)
        if args.orig_only:
            probes = [probe for probe in probes if probe.get("probe") == "orig"]
            for probe in probes:
                probe["probe_count"] = 1
        probes_path = work_dir / "probes.jsonl"
        scores_path = work_dir / "probe_scores.jsonl"
        traces_path = work_dir / "prune_traces.jsonl"
        metrics_path = work_dir / "metrics.json"
        samples_path = work_dir / "sample_scores.jsonl"
        write_jsonl(probes_path, probes)
        prune_config = LlavaPruneConfig(
            selector=args.selector,
            keep_ratio=args.keep_ratio,
            seed=args.seed,
            hybrid_core_ratio=args.hybrid_core_ratio,
            hybrid_context_ratio=args.hybrid_context_ratio,
            evidence_boost=args.evidence_boost,
            scope_alpha=args.scope_alpha,
            coin_alpha=args.coin_alpha,
            coin_beta=args.coin_beta,
            anchorprune_k_min=args.anchorprune_k_min,
            anchorprune_k_min_ratio=args.anchorprune_k_min_ratio,
            anchorprune_tau=args.anchorprune_tau,
            anchorprune_patience=args.anchorprune_patience,
            anchorprune_kmax_ratio=args.anchorprune_kmax_ratio,
            anchorprune_clip_model=args.anchorprune_clip_model,
            kept_indices_by_sample=_load_kept_indices(args.kept_indices, args.kept_indices_key) if args.kept_indices else None,
            position_mode=args.position_mode,
        )
        scores, traces = score_probes_with_llava_pruned(
            probes,
            pretrained=args.pretrained,
            prune_config=prune_config,
            revision=args.revision,
            device=args.device,
            device_map=args.device_map,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
            attn_implementation=args.attn_implementation or None,
            chat_template=args.chat_template or None,
            target_delimiter=args.target_delimiter,
            debug_forward=args.debug_forward,
            strict_images=not args.allow_missing_images,
        )
        write_jsonl(scores_path, scores)
        write_jsonl(traces_path, traces)
        result = aggregate_scores(scores)
        metrics = result["metrics"]
        metrics["pruning"] = summarize_llava_prune_traces(traces)
        metrics["prune_selector"] = args.selector
        metrics["prune_score_source"] = traces[0].get("score_source", "") if traces else ""
        kept_indices_path = getattr(args, "kept_indices", "")
        metrics["prune_budget_mode"] = "provided_indices" if kept_indices_path else "fixed"
        metrics["prune_target_keep_ratio"] = float(args.keep_ratio)
        metrics["prune_position_mode"] = args.position_mode
        metrics["prune_kept_indices_file"] = args.kept_indices
        metrics["prune_kept_indices_key"] = args.kept_indices_key if args.kept_indices else ""
        if str(args.selector).strip().lower() in {"scope", "official_scope"}:
            metrics["prune_scope_alpha"] = float(args.scope_alpha)
            metrics["prune_implementation_status"] = "official_algorithm_port"
        if str(args.selector).strip().lower() in {"coin", "paper_coin"}:
            metrics["prune_coin_alpha"] = float(args.coin_alpha)
            metrics["prune_coin_beta"] = float(args.coin_beta)
            metrics["prune_implementation_status"] = "paper_algorithm_port"
        if str(args.selector).strip().lower() in {"anchorprune", "official_anchorprune"}:
            metrics["prune_anchorprune_k_min"] = int(args.anchorprune_k_min)
            metrics["prune_anchorprune_k_min_ratio"] = float(args.anchorprune_k_min_ratio)
            metrics["prune_anchorprune_tau"] = float(args.anchorprune_tau)
            metrics["prune_anchorprune_patience"] = int(args.anchorprune_patience)
            metrics["prune_anchorprune_kmax_ratio"] = float(args.anchorprune_kmax_ratio)
            metrics["prune_anchorprune_clip_model"] = args.anchorprune_clip_model
            metrics["prune_anchorprune_commit"] = traces[0].get("anchorprune_commit", "") if traces else ""
            metrics["prune_implementation_status"] = "official_algorithm_port_exact_selector_parity"
        write_json(metrics_path, metrics)
        write_jsonl(samples_path, result["samples"])
        print(
            f"Wrote {len(probes)} probes, {len(scores)} LLaVA pruned probe scores, "
            f"{len(traces)} traces, and metrics to {work_dir}"
        )
        return

    if args.command == "run-llava-direct":
        from recap.llava_direct_backend import score_probes_with_llava_direct

        work_dir = Path(args.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        samples = read_jsonl(args.input)
        if args.limit is not None:
            samples = samples[: args.limit]
        probes = samples if args.is_probes else build_probe_dataset(samples, require_left_right=not args.keep_non_left_right, probe_mode=args.probe_mode)
        if args.orig_only:
            probes = [probe for probe in probes if probe.get("probe") == "orig"]
            for probe in probes:
                probe["probe_count"] = 1
        probes_path = work_dir / "probes.jsonl"
        scores_path = work_dir / "probe_scores.jsonl"
        metrics_path = work_dir / "metrics.json"
        samples_path = work_dir / "sample_scores.jsonl"
        write_jsonl(probes_path, probes)
        scores = score_probes_with_llava_direct(
            probes,
            pretrained=args.pretrained,
            revision=args.revision,
            device=args.device,
            device_map=args.device_map,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
            attn_implementation=args.attn_implementation or None,
            chat_template=args.chat_template or None,
            target_delimiter=args.target_delimiter,
            debug_forward=args.debug_forward,
            strict_images=not args.allow_missing_images,
        )
        write_jsonl(scores_path, scores)
        result = aggregate_scores(scores)
        write_json(metrics_path, result["metrics"])
        write_jsonl(samples_path, result["samples"])
        print(f"Wrote {len(probes)} probes, {len(scores)} LLaVA direct probe scores, and metrics to {work_dir}")
        return

    if args.command == "run-internvl-pruned":
        from recap.internvl_pruned_backend import (
            InternVLPruneConfig,
            score_probes_with_internvl_pruned,
            summarize_internvl_prune_traces,
        )

        work_dir = Path(args.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        samples = read_jsonl(args.input)
        if args.limit is not None:
            samples = samples[: args.limit]
        probes = samples if args.is_probes else build_probe_dataset(samples, require_left_right=not args.keep_non_left_right, probe_mode=args.probe_mode)
        if args.orig_only:
            probes = [probe for probe in probes if probe.get("probe") == "orig"]
            for probe in probes:
                probe["probe_count"] = 1
        probes_path = work_dir / "probes.jsonl"
        scores_path = work_dir / "probe_scores.jsonl"
        traces_path = work_dir / "prune_traces.jsonl"
        metrics_path = work_dir / "metrics.json"
        samples_path = work_dir / "sample_scores.jsonl"
        write_jsonl(probes_path, probes)
        prune_config = InternVLPruneConfig(
            selector=args.selector,
            keep_ratio=args.keep_ratio,
            seed=args.seed,
            hybrid_core_ratio=args.hybrid_core_ratio,
            hybrid_context_ratio=args.hybrid_context_ratio,
            evidence_boost=args.evidence_boost,
            kept_indices_by_sample=_load_kept_indices(args.kept_indices, args.kept_indices_key) if args.kept_indices else None,
            position_mode=args.position_mode,
        )
        scores, traces = score_probes_with_internvl_pruned(
            probes,
            pretrained=args.pretrained,
            prune_config=prune_config,
            revision=args.revision,
            device=args.device,
            device_map=args.device_map,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
            low_cpu_mem_usage=args.low_cpu_mem_usage,
            attn_implementation=args.attn_implementation or None,
            min_patches=args.min_patches,
            max_patches=args.max_patches,
            target_delimiter=args.target_delimiter,
            debug_forward=args.debug_forward,
            strict_images=not args.allow_missing_images,
        )
        write_jsonl(scores_path, scores)
        write_jsonl(traces_path, traces)
        result = aggregate_scores(scores)
        metrics = result["metrics"]
        metrics["pruning"] = summarize_internvl_prune_traces(traces)
        metrics["prune_selector"] = args.selector
        metrics["prune_score_source"] = traces[0].get("score_source", "") if traces else ""
        metrics["prune_budget_mode"] = "provided_indices" if args.kept_indices else "fixed"
        metrics["prune_target_keep_ratio"] = float(args.keep_ratio)
        metrics["prune_position_mode"] = args.position_mode
        metrics["prune_min_patches"] = int(args.min_patches)
        metrics["prune_max_patches"] = int(args.max_patches)
        metrics["prune_kept_indices_file"] = args.kept_indices
        metrics["prune_kept_indices_key"] = args.kept_indices_key if args.kept_indices else ""
        write_json(metrics_path, metrics)
        write_jsonl(samples_path, result["samples"])
        print(
            f"Wrote {len(probes)} probes, {len(scores)} InternVL pruned probe scores, "
            f"{len(traces)} traces, and metrics to {work_dir}"
        )
        return

    if args.command == "run-internvl-direct":
        from recap.internvl_direct_backend import score_probes_with_internvl_direct

        work_dir = Path(args.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        samples = read_jsonl(args.input)
        if args.limit is not None:
            samples = samples[: args.limit]
        probes = samples if args.is_probes else build_probe_dataset(samples, require_left_right=not args.keep_non_left_right, probe_mode=args.probe_mode)
        if args.orig_only:
            probes = [probe for probe in probes if probe.get("probe") == "orig"]
            for probe in probes:
                probe["probe_count"] = 1
        probes_path = work_dir / "probes.jsonl"
        scores_path = work_dir / "probe_scores.jsonl"
        metrics_path = work_dir / "metrics.json"
        samples_path = work_dir / "sample_scores.jsonl"
        write_jsonl(probes_path, probes)
        scores = score_probes_with_internvl_direct(
            probes,
            pretrained=args.pretrained,
            revision=args.revision,
            device=args.device,
            device_map=args.device_map,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
            low_cpu_mem_usage=args.low_cpu_mem_usage,
            attn_implementation=args.attn_implementation or None,
            min_patches=args.min_patches,
            max_patches=args.max_patches,
            target_delimiter=args.target_delimiter,
            debug_forward=args.debug_forward,
            strict_images=not args.allow_missing_images,
        )
        write_jsonl(scores_path, scores)
        result = aggregate_scores(scores)
        write_json(metrics_path, result["metrics"])
        write_jsonl(samples_path, result["samples"])
        print(f"Wrote {len(probes)} probes, {len(scores)} InternVL direct probe scores, and metrics to {work_dir}")
        return


if __name__ == "__main__":
    main()
