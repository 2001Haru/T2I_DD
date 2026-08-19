import csv
import sys
import tempfile
from pathlib import Path

from PIL import Image


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from make_random_cluster_member_montages import (  # noqa: E402
    read_assignment,
    render_class_montage,
    sample_rows,
)


def test_sampling_is_deterministic_and_uses_distinct_real_members():
    rows = []
    for cluster_id in range(7):
        for image_id in range(6):
            rows.append(
                {
                    "synset": "n00000001",
                    "cluster_id": cluster_id,
                    "image_value": f"{cluster_id}_{image_id}.jpg",
                }
            )
    first, audit = sample_rows(rows, 123, 5, 5)
    second, _ = sample_rows(rows, 123, 5, 5)
    assert first == second
    assert len(first) == 25
    assert len({row["cluster_id"] for row in first}) == 5
    for cluster_id in {row["cluster_id"] for row in first}:
        members = [row["image_value"] for row in first if row["cluster_id"] == cluster_id]
        assert len(members) == len(set(members)) == 5
    assert audit[0]["clusters_eligible"] == 7


def test_assignment_schema_and_montage_render():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        csv_path = root / "assignments.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["synset", "assigned_cluster", "image_path"])
            writer.writeheader()
            for cluster_id in range(5):
                for image_id in range(5):
                    image_path = root / f"{cluster_id}_{image_id}.png"
                    Image.new("RGB", (20, 10), (cluster_id * 20, image_id * 20, 0)).save(image_path)
                    writer.writerow(
                        {
                            "synset": "n00000001",
                            "assigned_cluster": cluster_id,
                            "image_path": image_path,
                        }
                    )
        selected, _ = sample_rows(read_assignment(csv_path), 10, 5, 5)
        for row in selected:
            row["resolved_path"] = row["image_value"]
        output = root / "montage.jpg"
        render_class_montage("test", "n00000001", selected, output, 32)
        assert output.is_file()
        with Image.open(output) as image:
            assert image.width == 150 + 5 * 32 + 4 * 4


def test_nearest_mode_uses_all_clusters_and_smallest_distances():
    rows = []
    for cluster_id in range(3):
        for image_id in range(7):
            rows.append(
                {
                    "synset": "n00000001",
                    "cluster_id": cluster_id,
                    "image_value": f"{cluster_id}_{image_id}.jpg",
                    "center_distance": float(7 - image_id),
                }
            )
    selected, _ = sample_rows(
        rows, 123, 1, 5, cluster_selection="all", member_selection="nearest"
    )
    assert len(selected) == 15
    assert {row["cluster_id"] for row in selected} == {0, 1, 2}
    for cluster_id in range(3):
        distances = [
            row["center_distance"] for row in selected if row["cluster_id"] == cluster_id
        ]
        assert sorted(distances) == [1.0, 2.0, 3.0, 4.0, 5.0]
