import argparse
import csv
import hashlib
import html
import json
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Randomly sample real members from randomly sampled visual clusters "
            "and render one row per cluster for blind visual inspection."
        )
    )
    parser.add_argument("--assignment", action="append", required=True, metavar="NAME=CSV")
    parser.add_argument("--data-root", action="append", default=[], metavar="NAME=DIR")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--clusters-per-class", type=int, default=5)
    parser.add_argument("--images-per-cluster", type=int, default=5)
    parser.add_argument("--tile-size", type=int, default=224)
    return parser.parse_args()


def parse_named(values, option):
    output = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{option} expects NAME=PATH, got {value!r}")
        name, path = value.split("=", 1)
        if not name or name in output:
            raise ValueError(f"Invalid or duplicate name for {option}: {name!r}")
        output[name] = Path(path).resolve()
    return output


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(seed, *parts):
    payload = "\0".join([str(seed), *map(str, parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def read_assignment(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Assignment CSV is empty: {path}")
    fields = set(rows[0])
    cluster_field = next(
        (field for field in ("assigned_cluster", "cluster_index", "cluster_id") if field in fields),
        None,
    )
    synset_field = next(
        (field for field in ("synset", "class_key", "class_id") if field in fields),
        None,
    )
    image_field = next(
        (field for field in ("image_path", "path", "relative", "relative_path") if field in fields),
        None,
    )
    if not cluster_field or not synset_field or not image_field:
        raise ValueError(
            f"Unsupported assignment schema in {path}: {sorted(fields)}"
        )
    normalized = []
    for row in rows:
        normalized.append(
            {
                "synset": str(row[synset_field]),
                "cluster_id": int(float(row[cluster_field])),
                "image_value": str(row[image_field]),
            }
        )
    return normalized


def resolve_image(image_value, data_root):
    path = Path(image_value)
    candidates = [path]
    if data_root is not None:
        candidates.extend(
            [
                data_root / path,
                data_root / "train" / path,
                data_root / "train" / path.name,
            ]
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Cannot resolve assignment image {image_value!r}; tried "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def sample_rows(rows, seed, clusters_per_class, images_per_cluster):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["synset"], row["cluster_id"])].append(row)
    by_class = defaultdict(dict)
    for (synset, cluster_id), members in grouped.items():
        by_class[synset][cluster_id] = members

    selected = []
    audit = []
    for synset, clusters in sorted(by_class.items()):
        eligible = sorted(
            cluster_id
            for cluster_id, members in clusters.items()
            if len(members) >= images_per_cluster
        )
        if len(eligible) < clusters_per_class:
            raise RuntimeError(
                f"{synset} has only {len(eligible)} clusters with at least "
                f"{images_per_cluster} members; need {clusters_per_class}"
            )
        class_rng = random.Random(stable_seed(seed, synset, "clusters"))
        chosen_clusters = sorted(class_rng.sample(eligible, clusters_per_class))
        audit.append(
            {
                "synset": synset,
                "clusters_total": len(clusters),
                "clusters_eligible": len(eligible),
                "clusters_excluded_too_small": len(clusters) - len(eligible),
                "selected_cluster_ids": chosen_clusters,
            }
        )
        for row_index, cluster_id in enumerate(chosen_clusters, start=1):
            members = sorted(clusters[cluster_id], key=lambda row: row["image_value"])
            member_rng = random.Random(stable_seed(seed, synset, cluster_id, "members"))
            chosen_members = member_rng.sample(members, images_per_cluster)
            for column_index, member in enumerate(chosen_members, start=1):
                selected.append(
                    {
                        **member,
                        "row_index": row_index,
                        "column_index": column_index,
                        "cluster_size": len(members),
                    }
                )
    return selected, audit


def fit_with_letterbox(image, size):
    image = ImageOps.exif_transpose(image).convert("RGB")
    fitted = ImageOps.contain(image, (size, size), Image.Resampling.LANCZOS)
    tile = Image.new("RGB", (size, size), (242, 242, 242))
    tile.paste(fitted, ((size - fitted.width) // 2, (size - fitted.height) // 2))
    return tile


def render_class_montage(dataset, synset, rows, output_path, tile_size):
    rows = sorted(rows, key=lambda row: (row["row_index"], row["column_index"]))
    row_ids = sorted({row["row_index"] for row in rows})
    columns = max(row["column_index"] for row in rows)
    left = 150
    header = 52
    gap = 4
    width = left + columns * (tile_size + gap) - gap
    height = header + len(row_ids) * (tile_size + gap) - gap
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), f"{dataset}: {synset} | random clusters and random real members", fill="black")
    for row_id in row_ids:
        current = [row for row in rows if row["row_index"] == row_id]
        y = header + (row_id - 1) * (tile_size + gap)
        first = current[0]
        draw.text((8, y + 8), f"cluster {first['cluster_id']}", fill="black")
        draw.text((8, y + 29), f"n={first['cluster_size']}", fill=(70, 70, 70))
        for member in current:
            x = left + (member["column_index"] - 1) * (tile_size + gap)
            with Image.open(member["resolved_path"]) as image:
                canvas.paste(fit_with_letterbox(image, tile_size), (x, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_index(output_dir, entries):
    lines = [
        "<!doctype html><meta charset='utf-8'>",
        "<title>Random cluster-member montages</title>",
        "<style>body{font-family:sans-serif;margin:24px}img{max-width:100%;border:1px solid #bbb}h2{margin-top:36px}</style>",
        "<h1>Random cluster-member montages</h1>",
    ]
    for entry in entries:
        relative = entry["montage_path"]
        lines.append(f"<h2>{html.escape(entry['dataset'])}: {html.escape(entry['synset'])}</h2>")
        lines.append(f"<a href='{html.escape(relative)}'><img src='{html.escape(relative)}'></a>")
    (output_dir / "index.html").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    assignments = parse_named(args.assignment, "--assignment")
    data_roots = parse_named(args.data_root, "--data-root")
    unknown_roots = set(data_roots) - set(assignments)
    if unknown_roots:
        raise ValueError(f"Data roots without assignments: {sorted(unknown_roots)}")
    if args.clusters_per_class < 1 or args.images_per_cluster < 1:
        raise ValueError("Sampling counts must be positive")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format_version": 1,
        "seed": args.seed,
        "clusters_per_class": args.clusters_per_class,
        "images_per_cluster": args.images_per_cluster,
        "sampling": "uniform clusters among clusters with enough members; uniform members without replacement",
        "assignments": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in assignments.items()
        },
        "data_roots": {name: str(path) for name, path in data_roots.items()},
    }
    manifest_path = output_dir / "manifest.json"
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous != manifest:
            raise RuntimeError(f"Existing output uses a different configuration: {manifest_path}")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    all_selected = []
    all_audit = []
    index_entries = []
    for dataset, assignment_path in assignments.items():
        selected, audit = sample_rows(
            read_assignment(assignment_path),
            stable_seed(args.seed, dataset),
            args.clusters_per_class,
            args.images_per_cluster,
        )
        for row in selected:
            row["dataset"] = dataset
            row["resolved_path"] = str(resolve_image(row["image_value"], data_roots.get(dataset)))
        for row in audit:
            row["dataset"] = dataset
        by_class = defaultdict(list)
        for row in selected:
            by_class[row["synset"]].append(row)
        for synset, rows in sorted(by_class.items()):
            relative = Path("montages") / dataset / f"{synset}.jpg"
            render_class_montage(dataset, synset, rows, output_dir / relative, args.tile_size)
            index_entries.append(
                {"dataset": dataset, "synset": synset, "montage_path": relative.as_posix()}
            )
        all_selected.extend(selected)
        all_audit.extend(audit)

    csv_rows = [
        {
            "dataset": row["dataset"],
            "synset": row["synset"],
            "row_index": row["row_index"],
            "cluster_id": row["cluster_id"],
            "cluster_size": row["cluster_size"],
            "column_index": row["column_index"],
            "image_path": row["resolved_path"],
        }
        for row in all_selected
    ]
    write_csv(output_dir / "selected_members.csv", csv_rows)
    (output_dir / "sampling_audit.json").write_text(
        json.dumps(all_audit, indent=2) + "\n", encoding="utf-8"
    )
    write_index(output_dir, index_entries)
    print(f"Wrote {len(index_entries)} class montages to {output_dir}")


if __name__ == "__main__":
    main()
