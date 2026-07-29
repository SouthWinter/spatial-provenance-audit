import json
from pathlib import Path
from tempfile import TemporaryDirectory

from recap.aggregate import aggregate_scores
from recap.caption_parse import parse_spatial_caption
from recap.gsrbench import prepare_gsrbench
from recap.images import draw_relation_boxes, encode_image_data_url
import recap.whatsup as whatsup_module
from PIL import Image
from recap.probes import build_probes
from recap.relations import build_relation_question


def test_build_probes_adds_bbox_views_for_left_right_sample():
    sample = {
        "id": "one",
        "image": "dummy.png",
        "subject": "A",
        "object": "B",
        "relation": "left_of",
        "answer": "yes",
        "subject_bbox": [0.0, 0.0, 0.4, 1.0],
        "object_bbox": [0.6, 0.0, 1.0, 1.0],
        "bbox_source": "oracle",
    }
    probes = build_probes(sample)
    assert len(probes) == 10
    assert {probe["probe"] for probe in probes} == {
        "orig",
        "flip_mapped",
        "flip_unmapped",
        "contra",
        "text_only",
        "inverse_role",
        "role_reversal",
        "subject_masked",
        "object_masked",
        "crop",
    }
    assert [probe for probe in probes if probe["probe"] == "flip_mapped"][0]["relation"] == "right_of"


def test_orig_text_crop_probe_mode_is_focused_and_bbox_aware():
    sample = {
        "id": "one",
        "image": "dummy.png",
        "subject": "A",
        "object": "B",
        "relation": "left_of",
        "answer": "yes",
        "subject_bbox": [0.0, 0.0, 0.4, 1.0],
        "object_bbox": [0.6, 0.0, 1.0, 1.0],
    }
    probes = build_probes(sample, probe_mode="orig_text_crop")
    assert [probe["probe"] for probe in probes] == ["orig", "text_only", "crop"]
    assert [probe["rice_view"] for probe in probes] == ["original", "text_only", "crop"]

    sample.pop("subject_bbox")
    sample.pop("object_bbox")
    probes = build_probes(sample, probe_mode="orig_text_crop")
    assert [probe["probe"] for probe in probes] == ["orig", "text_only"]


def test_orig_text_crop_boxed_probe_mode_adds_boxed_prompt_only_when_bbox_exists():
    sample = {
        "id": "one",
        "image": "dummy.png",
        "subject": "A",
        "object": "B",
        "relation": "left_of",
        "answer": "yes",
        "subject_bbox": [0.0, 0.0, 0.4, 1.0],
        "object_bbox": [0.6, 0.0, 1.0, 1.0],
    }
    probes = build_probes(sample, probe_mode="orig_text_crop_boxed")
    assert [probe["probe"] for probe in probes] == ["orig", "text_only", "crop", "boxed"]
    boxed = probes[-1]
    assert boxed["rice_view"] == "boxed"
    assert "red box labeled A" in boxed["question"]
    assert "blue box labeled B" in boxed["question"]

    sample.pop("subject_bbox")
    sample.pop("object_bbox")
    probes = build_probes(sample, probe_mode="orig_text_crop_boxed")
    assert [probe["probe"] for probe in probes] == ["orig", "text_only"]


def test_aggregate_scores_ranks_equivariance_violation_as_error_risk():
    rows = []

    def add(sample_id, correct, flip_yes_loss, flip_no_loss, text_yes_loss):
        target = "yes" if correct else "no"
        contra_target = "no" if target == "yes" else "yes"
        base = {
            "sample_id": sample_id,
            "relation": "left_of",
            "base_relation": "left_of",
            "target_answer": target,
            "has_bbox": True,
            "base_has_bbox": True,
            "bbox_source": "oracle",
            "probe_count": 8,
        }
        rows.extend(
            [
                {**base, "probe": "orig", "rice_view": "original", "question": "", "yes_loss": 0.1, "no_loss": 4.0 if correct else 4.0},
                {**base, "probe": "flip_mapped", "rice_view": "flip_mapped", "question": "", "yes_loss": flip_yes_loss, "no_loss": flip_no_loss},
                {**base, "probe": "flip_unmapped", "rice_view": "flip_unmapped", "question": "", "yes_loss": flip_no_loss, "no_loss": flip_yes_loss},
                {**base, "probe": "contra", "rice_view": "original", "question": "", "target_answer": contra_target, "yes_loss": 4.0 if correct else 0.2, "no_loss": 0.1},
                {**base, "probe": "text_only", "rice_view": "text_only", "question": "", "yes_loss": text_yes_loss, "no_loss": 3.0},
                {**base, "probe": "subject_masked", "rice_view": "subject_masked", "question": "", "yes_loss": text_yes_loss, "no_loss": 3.0},
                {**base, "probe": "object_masked", "rice_view": "object_masked", "question": "", "yes_loss": text_yes_loss, "no_loss": 3.0},
                {**base, "probe": "crop", "rice_view": "crop", "question": "", "yes_loss": 0.1, "no_loss": 4.0},
            ]
        )

    add("ok", True, flip_yes_loss=0.2, flip_no_loss=3.8, text_yes_loss=4.0)
    add("bad", False, flip_yes_loss=4.0, flip_no_loss=0.2, text_yes_loss=0.2)

    result = aggregate_scores(rows)
    samples = {sample["sample_id"]: sample for sample in result["samples"]}
    assert samples["bad"]["v_eq"] > samples["ok"]["v_eq"]
    assert samples["bad"]["rice_risk"] > samples["ok"]["rice_risk"]
    assert result["metrics"]["rice_error_auroc"] == 1.0


def test_source_question_template_preserves_original_question():
    sample = {
        "source_question": "From the camera view, where is the mug relative to the plate?",
        "source_options": ["left of", "right of", "in front of", "behind"],
    }
    left_question = build_relation_question(sample, "left_of")
    right_question = build_relation_question(sample, "right_of")
    assert "where is the mug relative to the plate" in left_question
    assert 'Is the correct answer "left of"?' in left_question
    assert 'Is the correct answer "right of"?' in right_question


def test_prepare_gsrbench_uses_only_external_two_object_splits():
    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        (root / "coco_qa_two_obj.json").write_text(
            json.dumps([["1", "a cup is left of a plate", "a cup is right of a plate"]]),
            encoding="utf-8",
        )
        (root / "vg_qa_two_obj.json").write_text(
            json.dumps([["2", "a cat is above a laptop", "a cat is below a laptop"]]),
            encoding="utf-8",
        )

        samples = prepare_gsrbench(
            dataset="external_two_object",
            root_dir=str(root),
            metadata_only=True,
            relation_families={"left_right", "vertical"},
        )

    assert len(samples) == 4
    assert {sample["source_split_name"] for sample in samples} == {
        "COCO-Spatial-Two",
        "GQA-Spatial-Two",
    }
    assert all(sample["dataset"] == "GSR-Bench" for sample in samples)
    assert all(sample["evaluation_scope"] == "external_two_object" for sample in samples)
    assert all(sample["uses_grounding_annotations"] is False for sample in samples)
    assert all(sample["overlaps_main_whatsup_controlled"] is False for sample in samples)


def test_gsrbench_gqa_depth_wording_is_normalized():
    front = parse_spatial_caption("A dog to the front of a chair")
    behind = parse_spatial_caption("A dog to the behind of a chair")
    top = parse_spatial_caption("A dog to the top of a chair")
    bottom = parse_spatial_caption("A dog to the bottom of a chair")
    assert front is not None and front.relation == "in_front_of"
    assert behind is not None and behind.relation == "behind"
    assert top is not None and top.relation == "above"
    assert bottom is not None and bottom.relation == "below"


def test_gsrbench_reuses_existing_coco_without_downloading_archive():
    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        coco_val = root / "coco" / "val2017"
        coco_val.mkdir(parents=True)
        (coco_val / "000000000001.jpg").write_bytes(b"placeholder")
        downloads: list[str] = []

        def fake_gdown(file_id: str, output: Path) -> None:
            downloads.append(output.name)
            output.write_text(
                json.dumps([["1", "a cup is left of a plate", "a cup is right of a plate"]]),
                encoding="utf-8",
            )

        original_gdown = whatsup_module._gdown
        whatsup_module._gdown = fake_gdown
        try:
            samples = prepare_gsrbench(
                dataset="coco_spatial_two",
                root_dir=str(root / "metadata"),
                coco_root=str(root / "coco"),
                download=True,
                relation_families={"left_right"},
            )
        finally:
            whatsup_module._gdown = original_gdown

    assert len(samples) == 2
    assert downloads == ["coco_qa_two_obj.json"]
    assert all("val2017" in sample["image"] for sample in samples)


def test_qwen_image_data_url_encoding():
    encoded = encode_image_data_url(Image.new("RGB", (2, 2), color="red"))
    assert encoded.startswith("data:image/jpeg;base64,")
    assert len(encoded) > len("data:image/jpeg;base64,")


def test_draw_relation_boxes_marks_subject_red_and_object_blue():
    image = Image.new("RGB", (20, 20), color="white")
    boxed = draw_relation_boxes(image, [0.0, 0.0, 0.4, 0.4], [0.6, 0.6, 1.0, 1.0])
    assert boxed.getpixel((1, 1))[0] > 200
    assert boxed.getpixel((13, 13))[2] > 200


def test_prompt_self_consistency_uses_five_image_prompts():
    sample = {
        "id": "sc-one",
        "image": "dummy.png",
        "subject": "cup",
        "object": "plate",
        "relation": "left_of",
        "answer": "yes",
    }
    probes = build_probes(sample, probe_mode="prompt_sc")
    assert len(probes) == 5
    assert {probe["probe"] for probe in probes} == {
        "orig",
        "prompt_sc_1",
        "prompt_sc_2",
        "prompt_sc_3",
        "prompt_sc_4",
    }
    assert all(probe["rice_view"] == "original" for probe in probes)
    assert len({probe["question"] for probe in probes}) == 5


def test_prompt_self_consistency_risk_uses_weakest_paraphrase():
    rows = []
    for sample_id, margins in {
        "stable": (3.0, 2.5, 2.0, 1.5, 2.5),
        "unstable": (3.0, 2.5, -1.0, 2.0, 2.5),
    }.items():
        for probe, margin in zip(("orig", "prompt_sc_1", "prompt_sc_2", "prompt_sc_3", "prompt_sc_4"), margins):
            rows.append(
                {
                    "sample_id": sample_id,
                    "probe": probe,
                    "rice_view": "original",
                    "relation": "left_of",
                    "base_relation": "left_of",
                    "target_answer": "yes",
                    "probe_count": 5,
                    "yes_loss": 0.0,
                    "no_loss": margin,
                }
            )
    result = aggregate_scores(rows)
    samples = {sample["sample_id"]: sample for sample in result["samples"]}
    assert samples["unstable"]["prompt_sc_risk"] > samples["stable"]["prompt_sc_risk"]
    assert samples["unstable"]["prompt_sc_disagreement_risk"] == 1.0 / 4.0
    assert result["metrics"]["avg_inference_calls"] == 10.0
