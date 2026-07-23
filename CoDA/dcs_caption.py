"""Build VLCP-style DCS captions for CoDA representatives.

The expensive image-caption stage is sharded by torchrun and written
incrementally. The cheap build stage assigns every class image to its nearest
saved CoDA representative in VAE space, then applies VLCP's class-common-word
filter and cluster-local weighted caption selection.
"""

import argparse
import glob
import json
import os
import pickle
import re
from collections import Counter
from datetime import datetime, timezone

import numpy as np
from tqdm import tqdm

from cluster_caption import _generate_caption, _load_llava, _write_json


DEFAULT_INSTRUCTION = (
    "Describe the physical appearance of the {class_name} in the image. "
    "Include details about its shape, posture, color, and any distinct features."
)
TOKEN_PATTERN = re.compile(r"[A-Za-z]+")


def _read_lines(path):
    with open(path, "r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]


def load_class_info(spec, misc_dir, nclass=10, phase=0):
    all_classes = _read_lines(os.path.join(misc_dir, "class_indices.txt"))
    spec_files = {
        "imageA": "imagenet-a.txt",
        "imageB": "imagenet-b.txt",
        "imageC": "imagenet-c.txt",
        "imageD": "imagenet-d.txt",
        "imageE": "imagenet-e.txt",
    }
    if spec not in spec_files:
        raise ValueError(f"DCS transfer currently supports {sorted(spec_files)}, got {spec!r}.")
    selected = _read_lines(os.path.join(misc_dir, spec_files[spec]))
    selected = selected[nclass * max(phase, 0):nclass * (max(phase, 0) + 1)]
    names = _read_lines(os.path.join(misc_dir, "class_names.txt"))
    class_names = {
        class_id: names[all_classes.index(class_id)]
        for class_id in selected
    }
    return selected, class_names


def _load_pickle(path):
    with open(path, "rb") as file:
        return pickle.load(file)


def _flat_features(items):
    if isinstance(items, np.ndarray) and items.ndim == 2:
        return items.astype(np.float32, copy=False)

    flattened = []
    for item in items:
        if hasattr(item, "detach"):
            item = item.detach().cpu().numpy()
        flattened.append(np.asarray(item, dtype=np.float32).reshape(-1))
    return np.stack(flattened)


def load_feature_records(args):
    selected, class_names = load_class_info(
        args.spec, args.misc_dir, nclass=args.nclass, phase=args.phase
    )
    records = {}
    for local_label, class_id in enumerate(selected):
        chunk_id = local_label // 10
        chunk = _load_pickle(f"{args.features_cache_path}_{chunk_id}")
        features = chunk["features"].get(local_label)
        paths = chunk["paths"].get(local_label)
        if features is None or paths is None:
            raise KeyError(f"Missing feature cache for local class {local_label} ({class_id}).")
        features = _flat_features(features)
        if len(features) != len(paths):
            raise ValueError(
                f"Feature/path mismatch for {class_id}: {len(features)} vs {len(paths)}."
            )
        records[class_id] = {
            "class_name": class_names[class_id],
            "features": features,
            "paths": [os.path.abspath(path) for path in paths],
        }
    return selected, records


def _caption_config(args):
    return {
        "format_version": 1,
        "spec": args.spec,
        "model": os.path.abspath(args.model),
        "instruction_template": args.instruction,
        "max_new_tokens": args.max_new_tokens,
    }


def _read_jsonl(path, allow_incomplete_final=False):
    rows = []
    if not os.path.isfile(path):
        return rows
    with open(path, "r", encoding="utf-8") as file:
        lines = file.readlines()
        last_nonempty = max(
            (index for index, line in enumerate(lines, start=1) if line.strip()),
            default=0,
        )
        for line_number, line in enumerate(lines, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                if allow_incomplete_final and line_number == last_nonempty:
                    break
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from error
    return rows


def _repair_trailing_jsonl(path):
    """Discard only a partially written final JSONL record after interruption."""
    if not os.path.isfile(path):
        return
    with open(path, "r", encoding="utf-8") as file:
        lines = file.readlines()
    valid_lines = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            json.loads(line)
        except json.JSONDecodeError as error:
            if line_number != len(lines):
                raise ValueError(f"Invalid non-final JSONL record at {path}:{line_number}") from error
            print(f"Discarding interrupted final caption record from: {path}")
            break
        valid_lines.append(line if line.endswith("\n") else line + "\n")
    if len(valid_lines) != len([line for line in lines if line.strip()]):
        temporary = f"{path}.repair.tmp"
        with open(temporary, "w", encoding="utf-8") as file:
            file.writelines(valid_lines)
        os.replace(temporary, path)


def _validate_or_write_rank_config(path, expected):
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as file:
            actual = json.load(file)
        # Caption shards are independent of launch parallelism. Older caches
        # recorded world_size; ignore it so a two-GPU run can resume on one GPU.
        actual.pop("world_size", None)
        if actual != expected:
            raise ValueError(
                f"Caption cache configuration changed at {path}. "
                "Use a new --caption-cache-dir to avoid mixing captions."
            )
        return
    _write_json(path, expected)


def caption_images(args):
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    os.makedirs(args.caption_cache_dir, exist_ok=True)

    rank_stem = os.path.join(args.caption_cache_dir, f"captions.rank{rank}")
    config = _caption_config(args)
    _validate_or_write_rank_config(f"{rank_stem}.meta.json", config)
    output_path = f"{rank_stem}.jsonl"
    _repair_trailing_jsonl(output_path)
    all_completed_rows = []
    for shard_path in glob.glob(os.path.join(args.caption_cache_dir, "captions.rank*.jsonl")):
        all_completed_rows.extend(
            _read_jsonl(shard_path, allow_incomplete_final=True)
        )
    completed = {
        row["image_path"]
        for row in all_completed_rows
        if isinstance(row.get("caption"), str) and row["caption"].strip()
    }

    selected, records = load_feature_records(args)
    tasks = []
    for class_id in selected:
        class_name = records[class_id]["class_name"].split(",")[0].strip()
        for image_path in records[class_id]["paths"]:
            tasks.append((class_id, class_name, image_path))
    tasks = [
        task for index, task in enumerate(tasks)
        if index % world_size == rank and task[2] not in completed
    ]
    if not tasks:
        print(f"Rank {rank}: caption shard is already complete.")
        return

    device = f"cuda:{local_rank}" if args.device == "cuda" else args.device
    print(f"Rank {rank}: loading LLaVA on {device}; {len(tasks)} images remain.")
    processor, model, dtype = _load_llava(args.model, device)
    with open(output_path, "a", encoding="utf-8", buffering=1) as file:
        for class_id, class_name, image_path in tqdm(
            tasks, desc=f"Rank {rank} DCS captions"
        ):
            instruction = args.instruction.format(class_name=class_name)
            caption = _generate_caption(
                model=model,
                processor=processor,
                dtype=dtype,
                device=device,
                image_path=image_path,
                instruction=instruction,
                max_new_tokens=args.max_new_tokens,
            )
            row = {
                "class_id": class_id,
                "image_path": image_path,
                "caption": " ".join(caption.split()),
            }
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_stop_words():
    try:
        from nltk.corpus import stopwords
        return set(stopwords.words("english"))
    except LookupError as error:
        raise RuntimeError(
            "NLTK English stopwords are missing. Run: "
            "python -m nltk.downloader stopwords"
        ) from error


def tokenize(text):
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


def select_dcs_captions(
    captions, assignments, class_name, threshold, top_k, stop_words=None
):
    """Apply the text-selection rule used by VLCP's gen_prototype.py."""
    if len(captions) != len(assignments):
        raise ValueError("Caption and assignment counts differ.")
    stop_words = _load_stop_words() if stop_words is None else set(stop_words)
    class_tokens = set(tokenize(class_name))

    sentence_tokens = [tokenize(caption) for caption in captions]
    class_presence = Counter()
    for tokens in sentence_tokens:
        class_presence.update(set(tokens))
    minimum_count = threshold * len(captions)
    class_common = {
        word for word, count in class_presence.items()
        if count >= minimum_count and word not in stop_words
    }

    selected = {}
    diagnostics = {}
    for cluster_index in sorted(set(int(value) for value in assignments)):
        indices = np.flatnonzero(assignments == cluster_index).tolist()
        frequencies = Counter()
        for index in indices:
            frequencies.update(
                word for word in sentence_tokens[index]
                if word not in stop_words
                and word not in class_tokens
                and word not in class_common
            )
        weighted_words = frequencies.most_common(top_k)
        best_index = indices[0]
        best_score = -1
        for index in indices:
            token_set = set(sentence_tokens[index])
            score = sum(weight for word, weight in weighted_words if word in token_set)
            if score > best_score:
                best_index = index
                best_score = score
        selected[cluster_index] = captions[best_index]
        diagnostics[cluster_index] = {
            "member_count": len(indices),
            "selected_member_index": best_index,
            "selection_score": best_score,
            "top_words": [
                {"word": word, "frequency": frequency}
                for word, frequency in weighted_words
            ],
        }
    return selected, sorted(class_common), diagnostics


def _load_caption_cache(args, expected_paths):
    rows = []
    metadata_files = sorted(glob.glob(os.path.join(args.caption_cache_dir, "captions.rank*.meta.json")))
    caption_files = sorted(glob.glob(os.path.join(args.caption_cache_dir, "captions.rank*.jsonl")))
    if not metadata_files or not caption_files:
        raise FileNotFoundError(
            f"No completed caption shards found under {args.caption_cache_dir}."
        )
    expected_config = None
    for path in metadata_files:
        with open(path, "r", encoding="utf-8") as file:
            config = json.load(file)
        comparable = {key: value for key, value in config.items() if key != "world_size"}
        if expected_config is None:
            expected_config = comparable
        elif comparable != expected_config:
            raise ValueError(f"Mixed caption configurations found in {args.caption_cache_dir}.")
    for path in caption_files:
        rows.extend(_read_jsonl(path))

    captions = {}
    for row in rows:
        image_path = os.path.abspath(row["image_path"])
        caption = row["caption"].strip()
        previous = captions.get(image_path)
        if previous is not None and previous != caption:
            raise ValueError(f"Conflicting cached captions for {image_path}.")
        captions[image_path] = caption
    missing = sorted(set(expected_paths) - set(captions))
    if missing:
        raise ValueError(
            f"Caption cache is incomplete: {len(missing)} images are missing; "
            f"first missing path is {missing[0]}."
        )
    return captions, expected_config


def _center_path(args, chunk_id):
    base, extension = os.path.splitext(args.saved_clusters_base_name)
    return os.path.join(args.specific_cluster_dir, f"{base}_{chunk_id}{extension}")


def _trim_caption(caption, max_words):
    if max_words <= 0:
        return caption
    words = caption.split()
    return caption if len(words) <= max_words else " ".join(words[:max_words]).rstrip(" ,;:")


def build_dcs(args):
    selected, records = load_feature_records(args)
    all_paths = [
        path for class_id in selected for path in records[class_id]["paths"]
    ]
    captions_by_path, caption_config = _load_caption_cache(args, all_paths)
    payload = {
        "metadata": {
            "format_version": 3,
            "method": "vlcp_dcs_transfer",
            "spec": args.spec,
            "threshold": args.threshold,
            "top_k": args.top_k,
            "max_caption_words": args.max_caption_words,
            "cluster_assignment": "nearest_saved_coda_representative_in_sdxl_vae_space",
            "caption_config": caption_config,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        "captions": {},
        "raw_captions": {},
        "caption_inputs": {},
        "diagnostics": {},
    }
    center_cache = {}
    for local_label, class_id in enumerate(selected):
        chunk_id = local_label // 10
        if chunk_id not in center_cache:
            center_cache[chunk_id] = _load_pickle(_center_path(args, chunk_id))
        centers = _flat_features(center_cache[chunk_id][local_label])
        if len(centers) != args.ipc:
            raise ValueError(f"Expected {args.ipc} centers for {class_id}, found {len(centers)}.")
        features = records[class_id]["features"]
        distances = (
            np.sum(features * features, axis=1, keepdims=True)
            + np.sum(centers * centers, axis=1)[None, :]
            - 2.0 * features @ centers.T
        )
        assignments = np.argmin(distances, axis=1)
        empty = sorted(set(range(args.ipc)) - set(assignments.tolist()))
        if empty:
            raise ValueError(
                f"Nearest-center assignment produced empty clusters for {class_id}: {empty}."
            )

        paths = records[class_id]["paths"]
        captions = [captions_by_path[path] for path in paths]
        selected_captions, class_common, diagnostics = select_dcs_captions(
            captions=captions,
            assignments=assignments,
            class_name=records[class_id]["class_name"],
            threshold=args.threshold,
            top_k=args.top_k,
        )
        for shift in range(args.ipc):
            raw_caption = selected_captions[shift]
            selected_index = diagnostics[shift]["selected_member_index"]
            payload["raw_captions"].setdefault(class_id, {})[str(shift)] = raw_caption
            payload["captions"].setdefault(class_id, {})[str(shift)] = _trim_caption(
                raw_caption, args.max_caption_words
            )
            payload["caption_inputs"].setdefault(class_id, {})[str(shift)] = {
                "image_path": paths[selected_index],
                "source_paths": [paths[index] for index in np.flatnonzero(assignments == shift)],
            }
        payload["diagnostics"][class_id] = {
            "class_name": records[class_id]["class_name"],
            "class_common_words": class_common,
            "clusters": {str(key): value for key, value in diagnostics.items()},
        }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    _write_json(args.output, payload)
    print(f"Saved {len(selected) * args.ipc} VLCP-style DCS captions to: {args.output}")


def add_common_arguments(parser):
    parser.add_argument("--spec", required=True)
    parser.add_argument("--misc-dir", default="./misc")
    parser.add_argument("--nclass", type=int, default=10)
    parser.add_argument("--phase", type=int, default=0)
    parser.add_argument("--features-cache-path", required=True)
    parser.add_argument("--caption-cache-dir", required=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    caption_parser = subparsers.add_parser("caption")
    add_common_arguments(caption_parser)
    caption_parser.add_argument("--model", required=True)
    caption_parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    caption_parser.add_argument("--max-new-tokens", type=int, default=128)
    caption_parser.add_argument("--device", default="cuda")

    build_parser = subparsers.add_parser("build")
    add_common_arguments(build_parser)
    build_parser.add_argument("--specific-cluster-dir", required=True)
    build_parser.add_argument("--saved-clusters-base-name", required=True)
    build_parser.add_argument("--ipc", type=int, default=10)
    build_parser.add_argument("--threshold", type=float, default=0.7)
    build_parser.add_argument("--top-k", type=int, default=30)
    build_parser.add_argument("--max-caption-words", type=int, default=0)
    build_parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == "caption":
        caption_images(args)
    else:
        if not 0.0 <= args.threshold <= 1.0:
            raise ValueError("--threshold must be between 0 and 1.")
        if args.top_k < 1:
            raise ValueError("--top-k must be positive.")
        build_dcs(args)


if __name__ == "__main__":
    main()
