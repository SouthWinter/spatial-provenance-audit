#!/usr/bin/env python
"""Filter image-free control JSONL files to the fixed InternVL hash split."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.calibrate_yesno_thresholds import stable_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/textocr_image_free_controls/development"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/textocr_image_free_controls/development_test536"),
    )
    parser.add_argument("--split", choices=("dev", "test"), default="test")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, dict[str, Any]] = {}
    for source in sorted(args.input_dir.glob("*.jsonl")):
        rows = read_jsonl(source)
        selected = [
            row
            for row in rows
            if stable_split(str(row["image_id"]), dev_buckets=5) == args.split
        ]
        output = args.output_dir / source.name
        write_jsonl(output, selected)
        outputs[source.stem] = {
            "path": str(output.resolve()),
            "rows": len(selected),
            "images": len({str(row["image_id"]) for row in selected}),
            "sha256": sha256_file(output),
        }
    row_counts = {item["rows"] for item in outputs.values()}
    image_counts = {item["images"] for item in outputs.values()}
    if row_counts != {536} or image_counts != {268}:
        raise RuntimeError(
            f"Unexpected fixed split sizes: rows={row_counts}, images={image_counts}"
        )
    summary = {
        "source_dir": str(args.input_dir.resolve()),
        "split": args.split,
        "split_rule": "sha1(image_id)[:8] mod 10; test iff bucket >= 5",
        "outputs": outputs,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(outputs)} {args.split} controls to {args.output_dir}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
