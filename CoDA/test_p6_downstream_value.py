from prepare_p6_datasets import assemble
import numpy as np

from analyze_p6_class_relationships import (
    build_cross_seed_records,
    correlation,
    hierarchical_correlation,
    rankdata,
)
from summarize_p6_downstream_value import contrast_definitions, paired_contrast


def test_assemble_uses_neutral_filler(tmp_path):
    source = tmp_path / "source" / "n00000001"
    filler = tmp_path / "filler" / "n00000001"
    source.mkdir(parents=True)
    filler.mkdir(parents=True)
    for index in (0, 2):
        (source / f"{index}.png").write_bytes(f"source-{index}".encode())
    (filler / "1.png").write_bytes(b"neutral-filler")

    destination = tmp_path / "output"
    audit = assemble(source.parent, filler.parent, destination, ipc=3)

    assert sorted(path.name for path in (destination / "n00000001").glob("*.png")) == [
        "0.png", "1.png", "2.png"
    ]
    assert (destination / "n00000001" / "1.png").read_bytes() == b"neutral-filler"
    assert audit["n00000001"]["neutral_filler_indices"] == [1]


def payload(label, correct, shuffled, matched_label=None):
    values = {
        "label": label,
        "matched_label": label if matched_label is None else matched_label,
        "correct": correct,
        "shuffled": shuffled,
    }
    return {
        prompt: {
            "overall_top1": [score, score + 1],
            "runs": [
                {"training_seed": 0, "overall_top1": score, "classes": []},
                {"training_seed": 1, "overall_top1": score + 1, "classes": []},
            ],
        }
        for prompt, score in values.items()
    }


def test_three_way_contrast_sign():
    # Correct-minus-label effects are 1, 2, 3, and 7 respectively, so the
    # init x guidance x correct interaction is (7 - 2) - (3 - 1) = 3.
    effects = {"i0g0": 1, "i1g0": 2, "i0g1": 3, "i1g1": 7}
    results = {}
    for regime, effect in effects.items():
        cells = payload(10, 10 + effect, 10 + effect)
        for prompt, item in cells.items():
            results[("imageA", 0, regime, prompt)] = item
    terms = contrast_definitions()["init_x_guidance_x_correct"]
    assert paired_contrast(results, "imageA", [0], terms) == [3, 3]


def test_matched_template_separates_style_from_caption_content():
    results = {}
    for regime in ("i0g0", "i1g0", "i0g1", "i1g1"):
        for prompt, item in payload(10, 16, 16, matched_label=14).items():
            results[("imageA", 0, regime, prompt)] = item
    definitions = contrast_definitions()
    assert paired_contrast(
        results, "imageA", [0], definitions["i0g0_matched_minus_label"]
    ) == [4, 4]
    assert paired_contrast(
        results, "imageA", [0], definitions["i0g0_correct_minus_matched"]
    ) == [2, 2]


def test_relationship_correlations_handle_ties_and_direction():
    assert rankdata([3, 1, 1, 2]).tolist() == [4.0, 1.5, 1.5, 3.0]
    assert correlation([1, 2, 3, 4], [4, 3, 2, 1], "pearson") == -1.0
    assert correlation([1, 2, 2, 4], [4, 3, 3, 1], "spearman") == -1.0


def test_hierarchical_correlation_preserves_paired_negative_signal():
    records = []
    for spec in ("imageA", "imageB"):
        for class_index in range(4):
            for generation_seed in (0, 1):
                records.append({
                    "spec": spec,
                    "class_id": f"n{class_index}",
                    "class_key": f"{spec}:n{class_index}",
                    "class_name": str(class_index),
                    "x": float(class_index),
                    "y": float(10 - class_index),
                    "generation_seed": generation_seed,
                })
    result, points = hierarchical_correlation(
        records, "x", "y", "spearman", samples=100, permutations=100,
        rng=np.random.default_rng(7), expected="negative",
    )
    assert result["value"] == -1.0
    assert len(points) == 8
    assert result["bootstrap_ci_upper"] < 0


def test_cross_seed_records_do_not_reuse_baseline_in_gain():
    rows = [
        {
            "spec": "imageA", "class_id": "n1", "class_key": "imageA:n1",
            "class_name": "one", "generation_seed": 0,
            "i0g0_matched_label": 20.0, "i0g0_content_gain": 8.0,
            "i0g0_label": 19.0, "i0g0_raw_dcs_gain": 9.0,
        },
        {
            "spec": "imageA", "class_id": "n1", "class_key": "imageA:n1",
            "class_name": "one", "generation_seed": 1,
            "i0g0_matched_label": 30.0, "i0g0_content_gain": 4.0,
            "i0g0_label": 29.0, "i0g0_raw_dcs_gain": 5.0,
        },
    ]
    cross = build_cross_seed_records(rows)
    forward = cross["seed0_to_seed1"][0]
    reverse = cross["seed1_to_seed0"][0]
    assert (forward["cross_seed_matched_baseline"], forward["cross_seed_content_gain"]) == (20.0, 4.0)
    assert (reverse["cross_seed_matched_baseline"], reverse["cross_seed_content_gain"]) == (30.0, 8.0)
