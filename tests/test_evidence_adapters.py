import json
from pathlib import Path

from recap.coco_evidence import attach_coco_evidence
from recap.ocr_regions import prepare_ocr_region_samples
from recap.textocr_regions import prepare_textocr_region_samples


def test_attach_coco_evidence_matches_subject_object_boxes(tmp_path: Path):
    instances = {
        "images": [{"id": 7, "file_name": "000000000007.jpg", "width": 100, "height": 200}],
        "categories": [
            {"id": 1, "name": "dining table"},
            {"id": 2, "name": "refrigerator"},
        ],
        "annotations": [
            {"id": 1, "image_id": 7, "category_id": 1, "bbox": [10, 20, 30, 40], "area": 1200, "iscrowd": 0},
            {"id": 2, "image_id": 7, "category_id": 2, "bbox": [50, 60, 20, 20], "area": 400, "iscrowd": 0},
        ],
    }
    instances_path = tmp_path / "instances.json"
    instances_path.write_text(json.dumps(instances), encoding="utf-8")
    samples = [
        {
            "id": "one",
            "image_id": "7",
            "subject": "photo of a dining table",
            "object": "fridge",
        }
    ]

    enriched, report = attach_coco_evidence(samples, instances_file=instances_path)

    assert report["matched_both"] == 1
    assert enriched[0]["bbox_source"] == "coco_instances"
    assert enriched[0]["subject_bbox"] == [0.1, 0.1, 0.4, 0.3]
    assert enriched[0]["object_bbox"] == [0.5, 0.3, 0.7, 0.4]
    assert len(enriched[0]["evidence_regions"]) == 2


def test_prepare_ocr_region_samples_accepts_token_boxes(tmp_path: Path):
    rows = [
        {
            "id": "ocr-one",
            "image": "doc.png",
            "question": "What word is shown?",
            "answer": "total",
            "width": 200,
            "height": 100,
            "ocr_tokens": [{"text": "total", "bbox": [10, 20, 50, 40]}],
        }
    ]
    input_path = tmp_path / "ocr.jsonl"
    input_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    samples = prepare_ocr_region_samples(input_path=input_path, image_root=tmp_path, dataset_name="ToyOCR")

    assert len(samples) == 1
    assert samples[0]["dataset"] == "ToyOCR"
    assert samples[0]["bbox_source"] == "ocr_regions"
    assert samples[0]["evidence_regions"] == [[0.05, 0.2, 0.25, 0.4]]


def test_prepare_textocr_region_samples_converts_word_annotations(tmp_path: Path):
    payload = {
        "imgs": {
            "img1": {"id": "img1", "width": 100, "height": 50, "set": "val", "file_name": "train/img1.jpg"}
        },
        "anns": {
            "ann1": {
                "id": "ann1",
                "image_id": "img1",
                "bbox": [10, 5, 20, 10],
                "utf8_string": "OPEN",
                "area": 200,
            }
        },
        "imgToAnns": {"img1": ["ann1"]},
    }
    annotation_path = tmp_path / "TextOCR_0.1_val.json"
    annotation_path.write_text(json.dumps(payload), encoding="utf-8")

    samples = prepare_textocr_region_samples(annotation_file=annotation_path, image_root=tmp_path)

    assert len(samples) == 1
    assert samples[0]["dataset"] == "TextOCR"
    assert samples[0]["bbox_source"] == "textocr_word_annotations"
    assert samples[0]["evidence_regions"] == [[0.1, 0.1, 0.3, 0.3]]
    assert samples[0]["ocr_tokens"] == [{"text": "OPEN", "bbox": [0.1, 0.1, 0.3, 0.3]}]
