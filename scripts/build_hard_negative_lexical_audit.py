#!/usr/bin/env python
"""Audit TextOCR-Hard near-miss negatives under stricter lexical normalizers."""

from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
PROBE_PATH = ROOT / "data" / "textocr_val_hard_probes_500img.jsonl"
REGION_PATH = ROOT / "data" / "textocr_val_regions_500.jsonl"
FULL_REGION_PATH = ROOT / "data" / "textocr_val_regions_full.jsonl"
OUT_DIR = ROOT / "runs" / "problem_optimization_audit" / "hard_negative_lexical_audit"


NORMALIZERS: dict[str, Callable[[str], str]] = {
    "raw_trim": lambda s: str(s or "").strip(),
    "nfc": lambda s: unicodedata.normalize("NFC", str(s or "").strip()),
    "nfkc": lambda s: unicodedata.normalize("NFKC", str(s or "").strip()),
    "nfkc_casefold": lambda s: unicodedata.normalize("NFKC", str(s or "").strip()).casefold(),
    "nfkc_casefold_nospace": lambda s: re.sub(
        r"\s+", "", unicodedata.normalize("NFKC", str(s or "").strip()).casefold()
    ),
    "nfkc_casefold_alnum": lambda s: "".join(
        ch for ch in unicodedata.normalize("NFKC", str(s or "").strip()).casefold() if ch.isalnum()
    ),
    "ascii_fold_alnum": lambda s: "".join(
        ch
        for ch in unicodedata.normalize("NFKD", str(s or "").strip())
        .encode("ascii", "ignore")
        .decode("ascii")
        .casefold()
        if ch.isalnum()
    ),
}


SUSPICIOUS_FIELDS = [
    "sample_id",
    "image_id",
    "source_text",
    "target_text",
    "reason",
    "edit_distance",
    "edit_type",
    "changed_char_class",
    "token_area_rank",
    "same_image_collision_normalizers",
    "source_target_same_normalizers",
    "matching_same_image_tokens",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-path", type=Path, default=PROBE_PATH)
    parser.add_argument("--region-path", type=Path, default=REGION_PATH)
    parser.add_argument("--full-region-path", type=Path, default=FULL_REGION_PATH)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    probes = [row for row in read_jsonl(args.probe_path) if row.get("binary_polarity") == "negative"]
    regions = {str(row["image_id"]): row for row in read_jsonl(args.region_path)}
    full_regions = read_jsonl(args.full_region_path) if args.full_region_path.exists() else list(regions.values())

    same_image_indexes = {
        image_id: build_token_index(region_row.get("ocr_tokens", [])) for image_id, region_row in regions.items()
    }
    global_indexes, global_image_sets = build_global_indexes(full_regions)

    detail_rows: list[dict[str, Any]] = []
    suspicious_rows: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    edit_type_counter: Counter[str] = Counter()
    changed_class_counter: Counter[str] = Counter()
    target_shape_counter: Counter[str] = Counter()
    rank_bucket_counter: Counter[str] = Counter()
    difficulty_counter: Counter[str] = Counter()
    edit_distances: list[int] = []
    norm_edit_distances: list[float] = []

    for probe in probes:
        source = clean_text(probe.get("source_text", ""))
        target = clean_text(probe.get("target_text", ""))
        image_id = str(probe.get("image_id", ""))
        edit_distance = levenshtein(NORMALIZERS["nfkc_casefold"](source), NORMALIZERS["nfkc_casefold"](target))
        normalized_edit_distance = edit_distance / max(1, len(NORMALIZERS["nfkc_casefold"](source)))
        edit_type = classify_edit(source, target, edit_distance)
        changed_class = changed_char_class(source, target)
        target_shape = text_shape(target)
        rank_bucket = area_rank_bucket(probe.get("token_area_rank", ""))
        difficulty = difficulty_bucket(normalized_edit_distance)
        edit_distances.append(edit_distance)
        norm_edit_distances.append(normalized_edit_distance)
        edit_type_counter[edit_type] += 1
        changed_class_counter[changed_class] += 1
        target_shape_counter[target_shape] += 1
        rank_bucket_counter[rank_bucket] += 1
        difficulty_counter[difficulty] += 1

        same_index = same_image_indexes.get(image_id, {})
        same_image_collision_normalizers = []
        matching_same_image_tokens: set[str] = set()
        for name, norm in NORMALIZERS.items():
            norm_target = norm(target)
            if norm_target and norm_target in same_index.get(name, {}):
                same_image_collision_normalizers.append(name)
                matching_same_image_tokens.update(same_index[name][norm_target])
                counters[f"same_image_collision_{name}"] += 1

        source_target_same_normalizers = [
            name
            for name, norm in NORMALIZERS.items()
            if norm(source) != "" and norm(source) == norm(target)
        ]
        for name in source_target_same_normalizers:
            counters[f"source_target_same_{name}"] += 1

        global_nfkc_images = global_image_sets["nfkc_casefold"].get(
            NORMALIZERS["nfkc_casefold"](target), set()
        )
        global_alnum_images = global_image_sets["nfkc_casefold_alnum"].get(
            NORMALIZERS["nfkc_casefold_alnum"](target), set()
        )
        global_nfkc_other = sorted(global_nfkc_images - {image_id})
        global_alnum_other = sorted(global_alnum_images - {image_id})
        if global_nfkc_other:
            counters["global_other_image_nfkc_casefold_collision"] += 1
        if global_alnum_other:
            counters["global_other_image_alnum_collision"] += 1

        target_alnum_empty = NORMALIZERS["nfkc_casefold_alnum"](target) == ""
        if target_alnum_empty:
            counters["target_alnum_empty"] += 1
        if any(ord(ch) > 127 for ch in source + target):
            counters["source_or_target_non_ascii"] += 1
        if any(ch.isdigit() for ch in target):
            counters["target_has_digit"] += 1
        if any(unicodedata.category(ch).startswith("P") for ch in target):
            counters["target_has_punctuation"] += 1
        if any(ch.isalpha() for ch in target):
            counters["target_has_alpha"] += 1
        if edit_distance > 2:
            counters["edit_distance_gt2"] += 1
        if same_image_collision_normalizers:
            counters["same_image_collision_any_normalizer"] += 1
        if source_target_same_normalizers:
            counters["source_target_same_any_normalizer"] += 1

        detail = {
            "sample_id": probe.get("sample_id", ""),
            "image_id": image_id,
            "source_text": source,
            "target_text": target,
            "edit_distance": edit_distance,
            "normalized_edit_distance": fmt(normalized_edit_distance),
            "edit_type": edit_type,
            "changed_char_class": changed_class,
            "target_shape": target_shape,
            "difficulty_bucket": difficulty,
            "token_area_rank": probe.get("token_area_rank", ""),
            "token_area_rank_bucket": rank_bucket,
            "same_image_collision_normalizers": ";".join(same_image_collision_normalizers),
            "source_target_same_normalizers": ";".join(source_target_same_normalizers),
            "matching_same_image_tokens": ";".join(sorted(matching_same_image_tokens)),
            "global_other_image_nfkc_casefold_matches": len(global_nfkc_other),
            "global_other_image_alnum_matches": len(global_alnum_other),
        }
        detail_rows.append(detail)

        reasons = []
        if same_image_collision_normalizers:
            reasons.append("same_image_collision_after_normalization")
        if source_target_same_normalizers:
            reasons.append("source_target_same_after_normalization")
        if edit_distance > 2:
            reasons.append("edit_distance_gt2")
        if target_alnum_empty:
            reasons.append("target_empty_after_alnum_normalization")
        if global_nfkc_other and edit_distance <= 1:
            reasons.append("near_miss_is_real_ocr_token_elsewhere")
        if reasons:
            suspicious_rows.append(
                {
                    "sample_id": detail["sample_id"],
                    "image_id": image_id,
                    "source_text": source,
                    "target_text": target,
                    "reason": ";".join(reasons),
                    "edit_distance": edit_distance,
                    "edit_type": edit_type,
                    "changed_char_class": changed_class,
                    "token_area_rank": probe.get("token_area_rank", ""),
                    "same_image_collision_normalizers": detail["same_image_collision_normalizers"],
                    "source_target_same_normalizers": detail["source_target_same_normalizers"],
                    "matching_same_image_tokens": detail["matching_same_image_tokens"],
                }
            )

    summary_rows = build_summary_rows(
        probes,
        counters,
        edit_distances,
        norm_edit_distances,
        same_image_indexes,
        global_indexes,
        args.full_region_path,
    )
    edit_class_rows = counter_rows("edit_type", edit_type_counter, len(probes))
    edit_class_rows += counter_rows("changed_char_class", changed_class_counter, len(probes))
    edit_class_rows += counter_rows("target_shape", target_shape_counter, len(probes))
    edit_class_rows += counter_rows("token_area_rank_bucket", rank_bucket_counter, len(probes))
    edit_class_rows += counter_rows("difficulty_bucket", difficulty_counter, len(probes))

    write_csv(args.out_dir / "hard_negative_lexical_details.csv", detail_rows)
    write_csv(args.out_dir / "hard_negative_lexical_summary.csv", summary_rows)
    write_csv(args.out_dir / "hard_negative_edit_class_summary.csv", edit_class_rows)
    write_csv(args.out_dir / "hard_negative_suspicious_examples.csv", suspicious_rows, SUSPICIOUS_FIELDS)
    (args.out_dir / "hard_negative_lexical_report.md").write_text(
        markdown(summary_rows, edit_class_rows, suspicious_rows), encoding="utf-8"
    )
    print(f"Wrote {args.out_dir / 'hard_negative_lexical_report.md'}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def clean_text(value: Any) -> str:
    return str(value or "").strip()


def build_token_index(tokens: list[dict[str, Any]]) -> dict[str, dict[str, set[str]]]:
    index: dict[str, dict[str, set[str]]] = {name: defaultdict(set) for name in NORMALIZERS}
    for item in tokens:
        if not isinstance(item, dict):
            continue
        raw = clean_text(item.get("text", ""))
        if not raw:
            continue
        for name, norm in NORMALIZERS.items():
            normed = norm(raw)
            if normed:
                index[name][normed].add(raw)
    return index


def build_global_indexes(
    region_rows: list[dict[str, Any]],
) -> tuple[dict[str, set[str]], dict[str, dict[str, set[str]]]]:
    values: dict[str, set[str]] = {name: set() for name in NORMALIZERS}
    image_sets: dict[str, dict[str, set[str]]] = {name: defaultdict(set) for name in NORMALIZERS}
    for row in region_rows:
        image_id = str(row.get("image_id", ""))
        for item in row.get("ocr_tokens", []):
            if not isinstance(item, dict):
                continue
            raw = clean_text(item.get("text", ""))
            if not raw:
                continue
            for name, norm in NORMALIZERS.items():
                normed = norm(raw)
                if normed:
                    values[name].add(normed)
                    image_sets[name][normed].add(image_id)
    return values, image_sets


def build_summary_rows(
    probes: list[dict[str, Any]],
    counters: Counter[str],
    edit_distances: list[int],
    norm_edit_distances: list[float],
    same_image_indexes: dict[str, dict[str, dict[str, set[str]]]],
    global_indexes: dict[str, set[str]],
    full_region_path: Path,
) -> list[dict[str, Any]]:
    n = len(probes)
    rows: list[dict[str, Any]] = [
        {"metric": "negative_probe_count", "value": n},
        {"metric": "image_count", "value": len({str(row.get("image_id", "")) for row in probes})},
        {"metric": "same_image_index_image_count", "value": len(same_image_indexes)},
        {"metric": "global_region_image_count", "value": global_index_image_note(full_region_path)},
        {"metric": "mean_edit_distance_nfkc_casefold", "value": fmt(mean_float(edit_distances))},
        {"metric": "max_edit_distance_nfkc_casefold", "value": max(edit_distances) if edit_distances else 0},
        {"metric": "mean_normalized_edit_distance_nfkc_casefold", "value": fmt(mean_float(norm_edit_distances))},
        {"metric": "global_unique_nfkc_casefold_token_count", "value": len(global_indexes["nfkc_casefold"])},
        {"metric": "global_unique_alnum_token_count", "value": len(global_indexes["nfkc_casefold_alnum"])},
    ]
    metrics = [
        "same_image_collision_raw_trim",
        "same_image_collision_nfc",
        "same_image_collision_nfkc",
        "same_image_collision_nfkc_casefold",
        "same_image_collision_nfkc_casefold_nospace",
        "same_image_collision_nfkc_casefold_alnum",
        "same_image_collision_ascii_fold_alnum",
        "same_image_collision_any_normalizer",
        "source_target_same_nfkc_casefold",
        "source_target_same_nfkc_casefold_alnum",
        "source_target_same_ascii_fold_alnum",
        "source_target_same_any_normalizer",
        "global_other_image_nfkc_casefold_collision",
        "global_other_image_alnum_collision",
        "edit_distance_gt2",
        "target_alnum_empty",
        "source_or_target_non_ascii",
        "target_has_alpha",
        "target_has_digit",
        "target_has_punctuation",
    ]
    for metric in metrics:
        count = counters.get(metric, 0)
        rows.append({"metric": f"{metric}_count", "value": count})
        rows.append({"metric": f"{metric}_rate", "value": fmt(count / max(1, n))})
    return rows


def global_index_image_note(full_region_path: Path) -> str:
    if not full_region_path.exists():
        return "full_region_file_missing"
    count = 0
    with full_region_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return str(count)


def counter_rows(group: str, counter: Counter[str], total: int) -> list[dict[str, Any]]:
    return [
        {"group": group, "bucket": key, "count": count, "rate": fmt(count / max(1, total))}
        for key, count in sorted(counter.items())
    ]


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + int(ca != cb)))
        prev = curr
    return prev[-1]


def classify_edit(source: str, target: str, dist: int) -> str:
    s = NORMALIZERS["nfkc_casefold"](source)
    t = NORMALIZERS["nfkc_casefold"](target)
    if dist == 0:
        return "identical_after_nfkc_casefold"
    if len(s) == len(t) and dist == 1:
        return "single_substitution"
    if len(s) == len(t) + 1 and dist == 1:
        return "single_deletion"
    if len(s) + 1 == len(t) and dist == 1:
        return "single_insertion"
    if dist == 1:
        return "single_edit_other"
    return "multi_edit_or_confusable_sequence"


def changed_char_class(source: str, target: str) -> str:
    s = NORMALIZERS["nfkc_casefold"](source)
    t = NORMALIZERS["nfkc_casefold"](target)
    classes = set()
    max_len = max(len(s), len(t))
    for i in range(max_len):
        cs = s[i] if i < len(s) else ""
        ct = t[i] if i < len(t) else ""
        if cs == ct:
            continue
        classes.add(char_class(cs))
        classes.add(char_class(ct))
    classes.discard("empty")
    if not classes:
        return "none"
    if len(classes) == 1:
        return next(iter(classes))
    return "+".join(sorted(classes))


def char_class(ch: str) -> str:
    if not ch:
        return "empty"
    if ch.isdigit():
        return "digit"
    if ch.isalpha():
        return "letter"
    if ord(ch) > 127:
        return "non_ascii"
    if unicodedata.category(ch).startswith("P"):
        return "punctuation"
    return "other"


def text_shape(value: str) -> str:
    has_alpha = any(ch.isalpha() for ch in value)
    has_digit = any(ch.isdigit() for ch in value)
    has_punct = any(unicodedata.category(ch).startswith("P") for ch in value)
    has_non_ascii = any(ord(ch) > 127 for ch in value)
    parts = []
    if has_alpha:
        parts.append("alpha")
    if has_digit:
        parts.append("digit")
    if has_punct:
        parts.append("punct")
    if has_non_ascii:
        parts.append("non_ascii")
    return "+".join(parts) if parts else "other"


def area_rank_bucket(value: Any) -> str:
    try:
        rank = int(float(value))
    except Exception:
        return "unknown"
    if rank <= 1:
        return "rank_1"
    if rank <= 5:
        return "rank_2_to_5"
    if rank <= 10:
        return "rank_6_to_10"
    return "rank_gt_10"


def difficulty_bucket(value: float) -> str:
    if value <= 0.10:
        return "norm_edit_le_0p10"
    if value <= 0.20:
        return "norm_edit_0p10_to_0p20"
    if value <= 0.35:
        return "norm_edit_0p20_to_0p35"
    return "norm_edit_gt_0p35"


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        if not fieldnames:
            return
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def markdown(
    summary_rows: list[dict[str, Any]],
    edit_class_rows: list[dict[str, Any]],
    suspicious_rows: list[dict[str, Any]],
) -> str:
    lines = [
        "# Hard-Negative Lexical Audit",
        "",
        "This audit checks whether TextOCR-Hard near-miss negative strings become accidental positives under common lexical normalizers.",
        "",
        "## Summary",
        "",
        md_table(summary_rows),
        "",
        "## Edit And Shape Distribution",
        "",
        md_table(edit_class_rows),
        "",
        "## Suspicious Examples",
        "",
        md_table(suspicious_rows[:50]) if suspicious_rows else "_No suspicious examples under the configured checks._",
        "",
        "## Interpretation",
        "",
        "- Same-image normalized collisions are construction risks because the negative target may actually be visible in the image.",
        "- Other-image global collisions are not construction errors; they only show that a near-miss string is a plausible OCR token elsewhere in TextOCR.",
        "- This audit is lexical and automatic. It does not replace manual human inspection.",
    ]
    return "\n".join(lines) + "\n"


def md_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_No rows._"
    cols = list(rows[0].keys())
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(escape_md(row.get(col, "")) for col in cols) + " |")
    return "\n".join(out)


def escape_md(value: Any) -> str:
    return str(value).replace("|", "\\|")


def mean_float(values: list[float] | list[int]) -> float:
    return sum(float(v) for v in values) / len(values) if values else 0.0


def fmt(value: Any) -> str:
    try:
        numeric = float(value)
        if abs(numeric) < 0.0005:
            numeric = 0.0
        return f"{numeric:.3f}"
    except Exception:
        return str(value)


if __name__ == "__main__":
    main()
