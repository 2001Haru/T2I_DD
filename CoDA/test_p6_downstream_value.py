from prepare_p6_datasets import assemble
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
