import csv
from pathlib import Path

import pytest

from analyze_p4_visual_interaction import build_paired_interactions, summarize_interactions


def make_row(mode, seed, target, pull, spec="imageA"):
    return {
        "encoder": "dino", "probe": "nearest_centroid", "spec": spec,
        "class_key": f"{spec}:n1", "class_id": "n1", "class_name": "class",
        "visual_cluster_id": "0", "caption_source_cluster_id": "1",
        "generation_seed": str(seed), "image_seed": str(100 + seed),
        "visual_mode": mode, "delta_target": str(target), "delta_pull": str(pull),
        "caption_rank_improvement": "1", "visual_rank_drop": "2",
    }


def test_builds_exact_visual_mode_interaction():
    rows = [make_row("i0g0", 0, 0.5, 0.3), make_row("i1g0", 0, 0.2, 0.1)]
    result = build_paired_interactions(rows)
    assert len(result) == 1
    assert result[0]["interaction_delta_target"] == pytest.approx(-0.3)
    assert result[0]["interaction_delta_pull"] == pytest.approx(-0.2)


def test_rejects_incomplete_pairs():
    with pytest.raises(ValueError, match="Incomplete visual-mode pair"):
        build_paired_interactions([make_row("i0g0", 0, 0.5, 0.3)])


def test_summary_averages_generation_seeds_before_bootstrap():
    rows = build_paired_interactions([
        make_row("i0g0", 0, 0.0, 0.0), make_row("i1g0", 0, 1.0, 1.0),
        make_row("i0g0", 1, 0.0, 0.0), make_row("i1g0", 1, 3.0, 3.0),
    ])
    summary = summarize_interactions(rows, samples=100, random_seed=7)
    target = next(row for row in summary if row["scope"] == "combined" and row["effect"] == "delta_target")
    assert target["mean"] == pytest.approx(2.0)
    assert target["class_cluster_groups"] == 1
    assert target["raw_paired_observations"] == 2
