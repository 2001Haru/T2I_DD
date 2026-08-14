import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "03_distiilation"))
from classes import IMAGENET2012_CLASSES  # noqa: E402


WORD_RE = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)?")
ATTRIBUTE_WORDS = {
    "alert", "angular", "arched", "bent", "black", "blue", "broad", "brown",
    "bushy", "circular", "compact", "curved", "dark", "dense", "erect", "fat",
    "flat", "fluffy", "furry", "gold", "golden", "gray", "green", "grey",
    "hairy", "large", "light", "long", "muscular", "narrow", "orange", "oval",
    "pale", "pointed", "red", "relaxed", "round", "rough", "shaggy", "short",
    "shiny", "slender", "small", "smooth", "spotted", "square", "standing",
    "striped", "sturdy", "thick", "thin", "upright", "white", "wide", "yellow",
}
ATTRIBUTE_SUFFIXES = (
    "ed", "ful", "ic", "ical", "ish", "ive", "less", "ous", "ular", "y"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit SD 1.5 caption truncation and caption class recoverability"
    )
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--dataset", action="append", default=[], metavar="NAME=CAPTIONS.jsonl")
    parser.add_argument("--dcs", action="append", default=[], metavar="NAME=DCS.json")
    parser.add_argument("--bank", action="append", default=[], metavar="NAME=BANK.json")
    parser.add_argument("--assignment", action="append", default=[], metavar="NAME=ASSIGNMENTS.csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--max-iter", type=int, default=3000)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--skip-probe", action="store_true")
    parser.add_argument("--force-features", action="store_true")
    return parser.parse_args()


def parse_named_paths(entries, flag):
    output = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"Expected NAME=PATH for {flag}, got: {entry}")
        name, value = entry.split("=", 1)
        name = name.strip()
        path = Path(value).expanduser()
        if not name or name in output:
            raise ValueError(f"Duplicate or empty dataset name for {flag}: {name}")
        if not path.is_file():
            raise FileNotFoundError(f"Missing {flag} path for {name}: {path}")
        output[name] = path
    return output


def read_jsonl_captions(dataset, path):
    rows = []
    seen = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            relative = str(item.get("file_name", item.get("image", ""))).replace("\\", "/")
            text = str(item.get("text", item.get("caption", ""))).strip()
            if not relative or not text:
                raise ValueError(f"Invalid caption row {path}:{line_number}")
            synset = Path(relative).parts[0]
            if synset not in IMAGENET2012_CLASSES:
                raise ValueError(f"Unknown synset in {path}:{line_number}: {synset}")
            if relative in seen:
                raise ValueError(f"Duplicate caption relative path in {path}: {relative}")
            seen.add(relative)
            rows.append({
                "dataset": dataset,
                "condition": "matched_caption",
                "record_id": relative,
                "relative": relative,
                "synset": synset,
                "cluster_id": "",
                "text": text,
            })
    return rows


def read_dcs(dataset, path, shift=1):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    correct = []
    shuffled = []
    for synset, captions in sorted(payload.items()):
        if synset not in IMAGENET2012_CLASSES or not captions:
            raise ValueError(f"Invalid DCS class in {path}: {synset}")
        for cluster_id, caption in enumerate(captions):
            text = str(caption).strip()
            if not text:
                raise ValueError(f"Empty DCS caption in {path}: {synset}/{cluster_id}")
            base = {
                "dataset": dataset,
                "record_id": f"{synset}:{cluster_id}",
                "relative": "",
                "synset": synset,
                "cluster_id": cluster_id,
                "text": text,
            }
            correct.append({**base, "condition": "correct_dcs"})
        for visual_cluster in range(len(captions)):
            source = (visual_cluster + shift) % len(captions)
            shuffled.append({
                "dataset": dataset,
                "condition": "shuffled_dcs",
                "record_id": f"{synset}:{visual_cluster}<-{source}",
                "relative": "",
                "synset": synset,
                "cluster_id": visual_cluster,
                "text": str(captions[source]).strip(),
            })
    return correct, shuffled


def read_bank(dataset, path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    classes = payload.get("classes", payload)
    rows = []
    for synset, entries in sorted(classes.items()):
        if synset not in IMAGENET2012_CLASSES or not isinstance(entries, list):
            raise ValueError(f"Invalid prompt-bank class in {path}: {synset}")
        for index, entry in enumerate(entries):
            text = str(entry.get("caption", entry.get("text", ""))).strip()
            if not text:
                raise ValueError(f"Empty prompt-bank caption in {path}: {synset}/{index}")
            rows.append({
                "dataset": dataset,
                "condition": "sparse_bank",
                "record_id": f"{synset}:{index}",
                "relative": str(entry.get("relative", "")).replace("\\", "/"),
                "synset": synset,
                "cluster_id": "",
                "text": text,
            })
    return rows


def label_rows(dataset, synsets):
    return [{
        "dataset": dataset,
        "condition": "label",
        "record_id": synset,
        "relative": "",
        "synset": synset,
        "cluster_id": "",
        "text": IMAGENET2012_CLASSES[synset],
    } for synset in sorted(synsets)]


def load_corpora(dataset_paths, dcs_paths, bank_paths):
    matched_by_dataset = {}
    all_rows = []
    for dataset, path in dataset_paths.items():
        matched = read_jsonl_captions(dataset, path)
        matched_by_dataset[dataset] = matched
        all_rows.extend(matched)
        all_rows.extend(label_rows(dataset, {row["synset"] for row in matched}))
        if dataset in dcs_paths:
            correct, shuffled = read_dcs(dataset, dcs_paths[dataset])
            if Counter(row["text"] for row in correct) != Counter(row["text"] for row in shuffled):
                raise RuntimeError(f"Correct/shuffled DCS marginal changed for {dataset}")
            all_rows.extend(correct)
            all_rows.extend(shuffled)
        if dataset in bank_paths:
            all_rows.extend(read_bank(dataset, bank_paths[dataset]))
    unknown = (set(dcs_paths) | set(bank_paths)) - set(dataset_paths)
    if unknown:
        raise ValueError(f"DCS/bank supplied without caption dataset: {sorted(unknown)}")
    return matched_by_dataset, all_rows


def content_ids(tokenizer, text, truncation=False):
    encoded = tokenizer(
        text,
        add_special_tokens=True,
        truncation=truncation,
        max_length=tokenizer.model_max_length if truncation else None,
    )["input_ids"]
    special = set(tokenizer.all_special_ids)
    return [int(token) for token in encoded if int(token) not in special], [int(token) for token in encoded]


def attribute_proxy_words(text):
    output = []
    for word in WORD_RE.findall(text.lower()):
        if word in ATTRIBUTE_WORDS or (len(word) >= 5 and word.endswith(ATTRIBUTE_SUFFIXES)):
            output.append(word)
    return output


def first_sentence(text):
    parts = re.split(r"(?<=[.!?])\s+", text.strip(), maxsplit=1)
    return parts[0] if parts else text


def audit_row(tokenizer, row):
    full_content, full_encoded = content_ids(tokenizer, row["text"], truncation=False)
    visible_content, visible_encoded = content_ids(tokenizer, row["text"], truncation=True)
    prefix_match = full_content[:len(visible_content)] == visible_content
    lost_ids = full_content[len(visible_content):] if prefix_match else []
    lost_text = tokenizer.decode(lost_ids, skip_special_tokens=True).strip() if lost_ids else ""
    attributes = attribute_proxy_words(row["text"])
    lost_attributes = attribute_proxy_words(lost_text)
    chunk = int(tokenizer.model_max_length)
    chunks = max(1, math.ceil(len(full_encoded) / chunk))
    first_content, _ = content_ids(tokenizer, first_sentence(row["text"]), truncation=False)
    return {
        **row,
        "content_tokens_full": len(full_content),
        "encoded_tokens_full": len(full_encoded),
        "train_visible_content_tokens": len(visible_content),
        "train_encoded_tokens": len(visible_encoded),
        "lost_content_tokens": max(0, len(full_content) - len(visible_content)),
        "lost_content_fraction": max(0, len(full_content) - len(visible_content)) / max(len(full_content), 1),
        "over_content_budget_75": int(len(full_content) > max(chunk - 2, 0)),
        "over_encoded_budget_77": int(len(full_encoded) > chunk),
        "inference_chunk_count": chunks,
        "attribute_proxy_total": len(attributes),
        "attribute_proxy_lost": len(lost_attributes),
        "attribute_proxy_lost_fraction": len(lost_attributes) / max(len(attributes), 1),
        "content_prefix_match": int(prefix_match),
        "first_sentence_content_tokens": len(first_content),
        "lost_tail": lost_text,
    }


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=np.float64), q)) if values else 0.0


def summarize_audit(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["condition"])].append(row)
    output = []
    for (dataset, condition), values in sorted(grouped.items()):
        tokens = [row["content_tokens_full"] for row in values]
        lost = [row["lost_content_tokens"] for row in values]
        attributes = sum(row["attribute_proxy_total"] for row in values)
        lost_attributes = sum(row["attribute_proxy_lost"] for row in values)
        output.append({
            "dataset": dataset,
            "condition": condition,
            "records": len(values),
            "content_tokens_mean": float(np.mean(tokens)),
            "content_tokens_median": percentile(tokens, 50),
            "content_tokens_p90": percentile(tokens, 90),
            "content_tokens_p95": percentile(tokens, 95),
            "content_tokens_max": max(tokens),
            "over_content_budget_75_fraction": float(np.mean([row["over_content_budget_75"] for row in values])),
            "over_encoded_budget_77_fraction": float(np.mean([row["over_encoded_budget_77"] for row in values])),
            "multi_chunk_fraction": float(np.mean([row["inference_chunk_count"] > 1 for row in values])),
            "chunk_count_mean": float(np.mean([row["inference_chunk_count"] for row in values])),
            "lost_content_tokens_mean": float(np.mean(lost)),
            "lost_content_token_fraction_micro": sum(lost) / max(sum(tokens), 1),
            "attribute_proxy_lost_fraction_micro": lost_attributes / max(attributes, 1),
            "first_sentence_content_tokens_median": percentile(
                [row["first_sentence_content_tokens"] for row in values], 50
            ),
            "content_prefix_match_fraction": float(np.mean([row["content_prefix_match"] for row in values])),
        })
    return output


def write_csv(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot_audit(rows, summary, output_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    datasets = sorted({row["dataset"] for row in rows})
    conditions = ["matched_caption", "correct_dcs", "sparse_bank", "label"]
    figure, axes = plt.subplots(len(datasets), 2, figsize=(14, 4.8 * len(datasets)), squeeze=False)
    for row_index, dataset in enumerate(datasets):
        axis = axes[row_index, 0]
        for condition in conditions:
            values = [
                row["content_tokens_full"] for row in rows
                if row["dataset"] == dataset and row["condition"] == condition
            ]
            if values:
                axis.hist(values, bins=35, alpha=0.45, density=True, label=condition)
        axis.axvline(75, color="black", linestyle="--", linewidth=1)
        axis.set_title(f"{dataset}: SD1.5 content-token lengths")
        axis.set_xlabel("content tokens (BOS/EOS excluded)")
        axis.set_ylabel("density")
        axis.legend(fontsize=8)

        axis = axes[row_index, 1]
        selected = [row for row in summary if row["dataset"] == dataset]
        labels = [row["condition"] for row in selected]
        values = [100 * row["over_encoded_budget_77_fraction"] for row in selected]
        axis.bar(range(len(selected)), values)
        axis.set_xticks(range(len(selected)), labels, rotation=25, ha="right")
        axis.set_ylim(0, 100)
        axis.set_ylabel("texts exceeding 77 encoded tokens (%)")
        axis.set_title(f"{dataset}: training-interface truncation")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def source_digest(rows, base_model):
    digest = hashlib.sha256(str(Path(base_model).resolve()).encode("utf-8"))
    for row in rows:
        digest.update(row["relative"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(row["text"].encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def mean_active(hidden, mask):
    return (hidden * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)


def encode_probe_features(rows, base_model, device, batch_size, local_files_only):
    import torch
    import torch.nn.functional as functional
    from transformers import CLIPTextModel, CLIPTokenizer

    target = torch.device(device)
    dtype = torch.float16 if target.type == "cuda" else torch.float32
    tokenizer = CLIPTokenizer.from_pretrained(
        base_model, subfolder="tokenizer", local_files_only=local_files_only
    )
    encoder = CLIPTextModel.from_pretrained(
        base_model, subfolder="text_encoder", local_files_only=local_files_only,
        torch_dtype=dtype,
    ).to(target)
    encoder.eval()
    texts = [row["text"] for row in rows]
    modes = {}
    with torch.inference_mode():
        truncated_mean = []
        truncated_pooled = []
        for start in range(0, len(texts), batch_size):
            encoded = tokenizer(
                texts[start:start + batch_size], padding="max_length",
                max_length=tokenizer.model_max_length, truncation=True,
                return_attention_mask=True, return_tensors="pt",
            )
            ids = encoded.input_ids.to(target)
            mask = encoded.attention_mask.to(target, dtype=torch.float32)
            output = encoder(ids)
            hidden = output.last_hidden_state.float()
            truncated_mean.append(
                functional.normalize(mean_active(hidden, mask), dim=1).cpu().numpy()
            )
            truncated_pooled.append(
                functional.normalize(output.pooler_output.float(), dim=1).cpu().numpy()
            )
        modes["train_t77_mean_hidden"] = np.concatenate(truncated_mean, axis=0)
        modes["train_t77_pooled"] = np.concatenate(truncated_pooled, axis=0)

        token_lengths = [
            len(tokenizer(text, truncation=False, add_special_tokens=True)["input_ids"])
            for text in texts
        ]
        chunks_by_index = defaultdict(list)
        chunk = tokenizer.model_max_length
        for index, length in enumerate(token_lengths):
            chunks_by_index[max(1, math.ceil(length / chunk))].append(index)
        full_mean = np.zeros_like(modes["train_t77_mean_hidden"])
        full_pooled = np.zeros_like(modes["train_t77_pooled"])
        for chunk_count, indices in sorted(chunks_by_index.items()):
            group_batch = max(1, batch_size // chunk_count)
            max_length = chunk_count * chunk
            for offset in range(0, len(indices), group_batch):
                selected = indices[offset:offset + group_batch]
                encoded = tokenizer(
                    [texts[index] for index in selected], padding="max_length",
                    max_length=max_length, truncation=False,
                    return_attention_mask=True, return_tensors="pt",
                )
                ids = encoded.input_ids.to(target)
                mask = encoded.attention_mask.to(target, dtype=torch.float32)
                batch = ids.shape[0]
                output = encoder(ids.reshape(batch * chunk_count, chunk))
                hidden = output.last_hidden_state.float().reshape(batch, max_length, -1)
                mean_features = functional.normalize(mean_active(hidden, mask), dim=1)
                pooled = output.pooler_output.float().reshape(batch, chunk_count, -1)
                chunk_weights = mask.reshape(batch, chunk_count, chunk).sum(dim=2)
                pooled_features = (
                    pooled * chunk_weights.unsqueeze(-1)
                ).sum(dim=1) / chunk_weights.sum(dim=1, keepdim=True).clamp_min(1)
                full_mean[np.asarray(selected)] = mean_features.cpu().numpy()
                full_pooled[np.asarray(selected)] = functional.normalize(
                    pooled_features, dim=1
                ).cpu().numpy()
        modes["inference_chunked_mean_hidden"] = full_mean
        modes["inference_chunked_pooled"] = full_pooled

    del encoder
    if target.type == "cuda":
        torch.cuda.empty_cache()
    return modes


def load_or_encode(rows, dataset, args, cache_dir):
    digest = source_digest(rows, args.base_model)
    cache_path = cache_dir / f"{dataset}_clip_text_features.npz"
    if cache_path.is_file() and not args.force_features:
        cached = np.load(cache_path, allow_pickle=False)
        if str(cached["digest"].item()) == digest:
            required = (
                "train_t77_mean_hidden", "train_t77_pooled",
                "inference_chunked_mean_hidden", "inference_chunked_pooled",
            )
            if all(name in cached.files for name in required):
                return {name: cached[name] for name in required}
    features = encode_probe_features(
        rows, args.base_model, args.device, args.batch_size, args.local_files_only
    )
    np.savez_compressed(cache_path, digest=digest, **features)
    return features


def evaluate_features(features, labels, folds, seed, max_iter):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, f1_score,
        normalized_mutual_info_score,
    )
    from sklearn.model_selection import StratifiedKFold

    labels = np.asarray(labels)
    counts = Counter(labels.tolist())
    if min(counts.values()) < folds:
        raise ValueError(f"Not enough samples for {folds}-fold CV: {counts}")
    predictions = np.empty(labels.shape, dtype=labels.dtype)
    fold_rows = []
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold, (train, test) in enumerate(splitter.split(features, labels)):
        model = LogisticRegression(
            max_iter=max_iter, class_weight="balanced", solver="lbfgs",
            random_state=seed + fold,
        )
        model.fit(features[train], labels[train])
        fold_predictions = model.predict(features[test])
        predictions[test] = fold_predictions
        fold_rows.append({
            "fold": fold,
            "samples": len(test),
            "top1": accuracy_score(labels[test], fold_predictions),
            "balanced_accuracy": balanced_accuracy_score(labels[test], fold_predictions),
            "macro_f1": f1_score(labels[test], fold_predictions, average="macro"),
            "normalized_mi": normalized_mutual_info_score(labels[test], fold_predictions),
        })
    summary = {
        "samples": len(labels),
        "classes": len(counts),
        "top1": accuracy_score(labels, predictions),
        "balanced_accuracy": balanced_accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro"),
        "normalized_mi": normalized_mutual_info_score(labels, predictions),
    }
    return summary, fold_rows


def normalize_relative(value):
    value = str(value).replace("\\", "/")
    marker = "/train/"
    if marker in value:
        value = value.split(marker, 1)[1]
    return value.lstrip("./")


def load_cluster_labels(path, rows):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        assignments = list(csv.DictReader(handle))
    if not assignments:
        return None, "assignment file is empty"
    fields = set(assignments[0])
    relative_field = next((field for field in ("relative", "relative_path", "image_path", "path") if field in fields), None)
    cluster_field = next((field for field in ("replay_cluster_id", "cluster_index", "cluster_id", "voronoi_cluster_id") if field in fields), None)
    if relative_field is None or cluster_field is None:
        return None, f"unsupported assignment columns: {sorted(fields)}"
    mapping = {
        normalize_relative(row[relative_field]): str(int(float(row[cluster_field])))
        for row in assignments if str(row.get(cluster_field, "")).strip()
    }
    labels = []
    missing = []
    for row in rows:
        key = normalize_relative(row["relative"])
        if key not in mapping:
            missing.append(key)
        else:
            labels.append(f"{row['synset']}:{mapping[key]}")
    if missing:
        return None, f"{len(missing)}/{len(rows)} caption rows absent from assignments"
    return labels, None


def run_probe(matched_by_dataset, assignment_paths, args, output_dir):
    cache_dir = output_dir / "feature_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    folds = []
    status = {}
    for dataset, rows in matched_by_dataset.items():
        features_by_mode = load_or_encode(rows, dataset, args, cache_dir)
        targets = {"class_id": [row["synset"] for row in rows]}
        if dataset in assignment_paths:
            cluster_labels, reason = load_cluster_labels(assignment_paths[dataset], rows)
            if cluster_labels is not None:
                targets["cluster_id_within_class"] = cluster_labels
                status[f"{dataset}:cluster_id_within_class"] = "evaluated"
            else:
                status[f"{dataset}:cluster_id_within_class"] = f"skipped: {reason}"
        else:
            status[f"{dataset}:cluster_id_within_class"] = "skipped: no --assignment supplied"
        for target_name, labels in targets.items():
            for mode, features in features_by_mode.items():
                try:
                    summary, fold_rows = evaluate_features(
                        features, labels, args.folds, args.seed, args.max_iter
                    )
                except ValueError as error:
                    status[f"{dataset}:{target_name}:{mode}"] = f"skipped: {error}"
                    continue
                summaries.append({
                    "dataset": dataset, "target": target_name,
                    "text_interface": mode, **summary,
                })
                folds.extend({
                    "dataset": dataset, "target": target_name,
                    "text_interface": mode, **row,
                } for row in fold_rows)
    write_csv(output_dir / "caption_probe_summary.csv", summaries)
    write_csv(output_dir / "caption_probe_folds.csv", folds)
    deltas = probe_interface_deltas(summaries)
    write_csv(output_dir / "caption_probe_interface_delta.csv", deltas)
    (output_dir / "caption_probe_status.json").write_text(
        json.dumps(status, indent=2, sort_keys=True), encoding="utf-8"
    )
    plot_probe(summaries, output_dir / "caption_probe.png")
    return summaries, status


def probe_interface_deltas(rows):
    metrics = ("top1", "balanced_accuracy", "macro_f1", "normalized_mi")
    output = []
    for dataset in sorted({row["dataset"] for row in rows}):
        for target in sorted({row["target"] for row in rows if row["dataset"] == dataset}):
            selected = {
                row["text_interface"]: row for row in rows
                if row["dataset"] == dataset and row["target"] == target
            }
            for representation in ("pooled", "mean_hidden"):
                train = selected.get(f"train_t77_{representation}")
                full = selected.get(f"inference_chunked_{representation}")
                if train is None or full is None:
                    continue
                for metric in metrics:
                    output.append({
                        "dataset": dataset,
                        "target": target,
                        "representation": representation,
                        "metric": metric,
                        "train_t77": train[metric],
                        "inference_chunked": full[metric],
                        "chunked_minus_t77": full[metric] - train[metric],
                    })
    return output


def plot_probe(rows, output_path):
    if not rows:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    class_rows = [row for row in rows if row["target"] == "class_id"]
    datasets = sorted({row["dataset"] for row in class_rows})
    metrics = ("top1", "macro_f1", "normalized_mi")
    figure, axes = plt.subplots(1, len(metrics), figsize=(15, 4.6))
    width = 0.36
    x = np.arange(len(datasets))
    for axis, metric in zip(axes, metrics):
        for offset, mode in (
            (-width / 2, "train_t77_pooled"),
            (width / 2, "inference_chunked_pooled"),
        ):
            values = [next(
                row[metric] for row in class_rows
                if row["dataset"] == dataset and row["text_interface"] == mode
            ) for dataset in datasets]
            axis.bar(x + offset, values, width, label=mode)
        axis.set_xticks(x, datasets)
        axis.set_ylim(0, 1)
        axis.set_title(metric)
    axes[0].legend()
    figure.suptitle("Caption class recoverability in the SD1.5 text encoder")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main():
    args = parse_args()
    if not args.dataset:
        raise ValueError("At least one --dataset NAME=CAPTIONS.jsonl is required")
    dataset_paths = parse_named_paths(args.dataset, "--dataset")
    dcs_paths = parse_named_paths(args.dcs, "--dcs")
    bank_paths = parse_named_paths(args.bank, "--bank")
    assignment_paths = parse_named_paths(args.assignment, "--assignment")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from transformers import CLIPTokenizer
    tokenizer = CLIPTokenizer.from_pretrained(
        args.base_model, subfolder="tokenizer", local_files_only=args.local_files_only
    )
    if tokenizer.model_max_length != 77:
        raise RuntimeError(f"Expected SD1.5 CLIP max length 77, got {tokenizer.model_max_length}")

    matched_by_dataset, corpus_rows = load_corpora(dataset_paths, dcs_paths, bank_paths)
    audited = [audit_row(tokenizer, row) for row in corpus_rows]
    summary = summarize_audit(audited)
    write_csv(output_dir / "token_audit_records.csv", audited)
    write_csv(output_dir / "token_audit_summary.csv", summary)
    truncated = sorted(
        (row for row in audited if row["lost_content_tokens"] > 0),
        key=lambda row: row["lost_content_tokens"], reverse=True,
    )[:200]
    write_csv(output_dir / "most_truncated_examples.csv", truncated)
    plot_audit(audited, summary, output_dir / "token_audit.png")

    probe_summary = []
    probe_status = {"probe": "skipped by --skip-probe"}
    if not args.skip_probe:
        probe_summary, probe_status = run_probe(
            matched_by_dataset, assignment_paths, args, output_dir
        )
    payload = {
        "format_version": 1,
        "base_model": str(Path(args.base_model).resolve()),
        "tokenizer_model_max_length": tokenizer.model_max_length,
        "training_interface": "single CLIP chunk with max_length=77 and truncation=True",
        "inference_interface": "untruncated CLIP IDs split into 77-token chunks",
        "probe_representation_boundary": (
            "Pooled probes use CLIP pooler_output; multi-chunk pooled features are active-token-"
            "weighted means of chunk pooler outputs. Mean-hidden probes are also reported and "
            "align more directly with the hidden states consumed by generation."
        ),
        "attribute_metric_boundary": (
            "attribute_proxy uses a fixed physical-attribute lexicon and adjective-like suffixes; "
            "lost content-token fraction is the primary truncation measure"
        ),
        "datasets": {name: len(rows) for name, rows in matched_by_dataset.items()},
        "heavy_truncation_rule": "flagged when more than 20% of a condition exceeds 77 encoded tokens",
        "heavy_truncation_conditions": [
            f"{row['dataset']}:{row['condition']}" for row in summary
            if row["over_encoded_budget_77_fraction"] > 0.20
        ],
        "token_audit": summary,
        "probe": probe_summary,
        "probe_status": probe_status,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"Caption-interface audit saved to: {output_dir}")


if __name__ == "__main__":
    main()
