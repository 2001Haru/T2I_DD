"""Offline VLM captioning for CoDA cluster representative images."""

import json
import os
import re
from datetime import datetime, timezone

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm


def _caption_file_payload(args):
    return {
        "metadata": {
            "format_version": 3,
            "model": args.cluster_caption_model_path,
            "instruction_template": args.cluster_caption_instruction,
            "max_new_tokens": args.cluster_caption_max_new_tokens,
            "max_words": args.cluster_caption_max_words,
            "image_mode": args.cluster_caption_image_mode,
            "neighbor_count": args.cluster_caption_neighbor_count,
            "montage_tile_size": args.cluster_caption_montage_tile_size,
            "neighbor_selection_space": "sdxl_vae_latent_mean_before_standardization",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        "captions": {},
        "raw_captions": {},
        "caption_inputs": {},
    }


def _write_json(path, payload):
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")
    os.replace(tmp_path, path)


def load_cluster_captions(path, sel_classes, ipc):
    """Load and validate captions for every representative image used by CoDA."""
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Cluster caption file was not found: {path}. "
            "Run with --generate_cluster_captions first."
        )

    with open(path, "r", encoding="utf-8") as file:
        payload = json.load(file)

    captions = payload.get("captions")
    if not isinstance(captions, dict):
        raise ValueError(f"Invalid caption file (missing 'captions' mapping): {path}")

    missing = []
    for class_id in sel_classes:
        class_captions = captions.get(class_id)
        if not isinstance(class_captions, dict):
            missing.append(f"{class_id}/*")
            continue
        for shift in range(ipc):
            caption = class_captions.get(str(shift))
            if not isinstance(caption, str) or not caption.strip():
                missing.append(f"{class_id}/{shift}")

    if missing:
        preview = ", ".join(missing[:10])
        suffix = " ..." if len(missing) > 10 else ""
        raise ValueError(
            f"Caption file is incomplete; missing {len(missing)} entries: {preview}{suffix}. "
            "Re-run with --generate_cluster_captions --overwrite_cluster_captions."
        )

    return captions


def _load_llava(model_path, device):
    try:
        from transformers import AutoProcessor, LlavaForConditionalGeneration
    except ImportError as error:
        raise ImportError(
            "LLaVA captioning requires transformers with LLaVA support. "
            "Install the project's requirements before running this stage."
        ) from error

    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    processor = AutoProcessor.from_pretrained(model_path)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    return processor, model, dtype


def _build_llava_prompt(processor, instruction):
    if getattr(processor, "chat_template", None):
        conversation = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": instruction},
            ],
        }]
        return processor.apply_chat_template(conversation, add_generation_prompt=True)
    return f"USER: <image>\n{instruction}\nASSISTANT:"


def _generate_caption(model, processor, dtype, device, image_path, instruction, max_new_tokens):
    prompt = _build_llava_prompt(processor, instruction)
    with Image.open(image_path) as image:
        inputs = processor(text=prompt, images=image.convert("RGB"), return_tensors="pt")

    inputs = {key: value.to(device) for key, value in inputs.items()}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype=dtype)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )

    prompt_length = inputs["input_ids"].shape[1]
    caption = processor.batch_decode(
        generated_ids[:, prompt_length:], skip_special_tokens=True, clean_up_tokenization_spaces=True
    )[0].strip()
    if not caption:
        raise RuntimeError(f"LLaVA returned an empty caption for {image_path}")
    return " ".join(caption.split())


_LAYOUT_PHRASES = (
    r"\b(?:in|across|among|from|throughout)\s+(?:all\s+)?(?:the\s+)?"
    r"(?:four|4|multiple|several)\s+(?:image\s+)?(?:tiles?|titles?|panels?|images?)\b",
    r"\b(?:in|across|among|from|throughout)\s+(?:these|the)\s+"
    r"(?:tiles?|titles?|panels?|images?|examples?)\b",
    r"\b(?:the\s+)?(?:four|4)\s+(?:tiles?|titles?|panels?|images?)\b",
    r"\b(?:each|every)\s+(?:tile|title|panel|image|example)\b",
)
_TRAILING_FILLER = re.compile(
    r"(?:\s+|,)+(?:and|or|with|including|featuring|such as|as well as)\s*$",
    flags=re.IGNORECASE,
)


def _remove_layout_references(text):
    text = re.sub(
        r"(^|[.!?]\s+)(?:the\s+)?(?:(?:four|4|these|provided|multiple)\s+)?"
        r"(?:images?|examples?|tiles?|titles?|panels?)\s+(?:show|depict|display|feature)\s+",
        r"\1", text, flags=re.IGNORECASE,
    )
    for pattern in _LAYOUT_PHRASES:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    return " ".join(text.split()).lstrip(" ,;:-")


def _is_meta_sentence(sentence):
    lowered = sentence.lower()
    return any(phrase in lowered for phrase in (
        "share a similar physical appearance",
        "shares a similar physical appearance",
        "shared physical appearance pattern",
        "common physical appearance pattern",
    ))


def _trim_words(text, max_words):
    words = text.split()
    if len(words) <= max_words:
        return text.strip()

    sentence_matches = list(re.finditer(r"[^.!?]+[.!?]", text))
    completed = []
    completed_words = 0
    for match in sentence_matches:
        sentence = match.group(0).strip()
        count = len(sentence.split())
        if completed_words + count > max_words:
            break
        completed.append(sentence)
        completed_words += count
    if completed:
        return " ".join(completed)

    trimmed = " ".join(words[:max_words]).rstrip(" ,;:")
    return _TRAILING_FILLER.sub("", trimmed).rstrip(" ,;:") + "."


def normalize_caption(caption, image_mode, max_words):
    """Convert VLM prose into a concise phrase safe for a single-image prompt."""
    normalized = " ".join(caption.split()).strip()
    if image_mode == "montage_neighbors":
        normalized = _remove_layout_references(normalized)
        sentences = [part.strip() for part in re.findall(r"[^.!?]+[.!?]?", normalized)]
        useful = [sentence for sentence in sentences if sentence and not _is_meta_sentence(sentence)]
        normalized = " ".join(useful)

    # Drop a final fragment caused by max_new_tokens when complete content precedes it.
    if normalized and normalized[-1] not in ".!?" and re.search(r"[.!?]", normalized):
        normalized = normalized[:max(match.end() for match in re.finditer(r"[.!?]", normalized))]
    normalized = _trim_words(normalized, max_words)
    normalized = _TRAILING_FILLER.sub("", normalized).strip(" ,;:")
    if not normalized:
        raise ValueError(f"Caption became empty after normalization: {caption!r}")
    return normalized


def _validate_caption_config(path, args):
    with open(path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    metadata = payload.get("metadata", {})
    expected = {
        "model": args.cluster_caption_model_path,
        "instruction_template": args.cluster_caption_instruction,
        "max_new_tokens": args.cluster_caption_max_new_tokens,
        "max_words": args.cluster_caption_max_words,
        "image_mode": args.cluster_caption_image_mode,
        "neighbor_count": args.cluster_caption_neighbor_count,
        "montage_tile_size": args.cluster_caption_montage_tile_size,
    }
    mismatches = [
        key for key, value in expected.items() if metadata.get(key) != value
    ]
    if mismatches:
        raise ValueError(
            f"Caption configuration changed for {path}: {', '.join(mismatches)}. "
            "Use a new --cluster_caption_file or pass --overwrite_cluster_captions."
        )


def _load_pickle(path):
    import pickle

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Required clustering artifact was not found: {path}")
    with open(path, "rb") as file:
        return pickle.load(file)


def _to_flat_numpy(items):
    if isinstance(items, np.ndarray) and items.ndim == 2:
        return items.astype(np.float32, copy=False)
    return np.stack([np.asarray(item, dtype=np.float32).reshape(-1) for item in items])


def _nearest_image_paths(features, paths, center, count):
    if count < 1:
        raise ValueError("Neighbor count must be positive.")
    features = _to_flat_numpy(features)
    center = np.asarray(center, dtype=np.float32).reshape(-1)
    if features.shape[1] != center.shape[0]:
        raise ValueError(
            f"Feature dimension mismatch: samples have {features.shape[1]}, center has {center.shape[0]}."
        )
    if len(paths) != len(features):
        raise ValueError(f"Feature/path count mismatch: {len(features)} features and {len(paths)} paths.")
    if count > len(paths):
        raise ValueError(f"Requested {count} caption neighbors, but the class has only {len(paths)} images.")

    differences = features - center
    squared_distances = np.einsum("ij,ij->i", differences, differences)
    candidates = np.argpartition(squared_distances, count - 1)[:count]
    nearest = candidates[np.lexsort((candidates, squared_distances[candidates]))]
    return [paths[index] for index in nearest], [float(squared_distances[index]) for index in nearest]


def _save_montage(image_paths, output_path, tile_size):
    columns = 2
    rows = (len(image_paths) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * tile_size, rows * tile_size), color="white")
    for index, image_path in enumerate(image_paths):
        with Image.open(image_path) as image:
            tile = image.convert("RGB").resize((tile_size, tile_size), Image.Resampling.LANCZOS)
        canvas.paste(tile, ((index % columns) * tile_size, (index // columns) * tile_size))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    canvas.save(output_path)


def _build_montage_tasks(args, sel_classes, class_id_to_name):
    tasks = []
    caption_inputs = {}
    chunk_cache = {}
    center_cache = {}

    for local_label, class_id in enumerate(sel_classes):
        chunk_id = local_label // 10
        if chunk_id not in chunk_cache:
            feature_path = f"{args.features_cache_path}_{chunk_id}"
            chunk_cache[chunk_id] = _load_pickle(feature_path)

            base, extension = os.path.splitext(args.saved_clusters_base_name)
            center_path = os.path.join(args.specific_cluster_dir, f"{base}_{chunk_id}{extension}")
            center_cache[chunk_id] = _load_pickle(center_path)

        class_features = chunk_cache[chunk_id]["features"].get(local_label)
        class_paths = chunk_cache[chunk_id]["paths"].get(local_label)
        class_centers = center_cache[chunk_id].get(local_label)
        if class_features is None or class_paths is None or class_centers is None:
            raise KeyError(f"Missing cached clustering data for local class {local_label} ({class_id}).")
        if len(class_centers) != args.IPC:
            raise ValueError(
                f"Expected {args.IPC} centers for {class_id}, found {len(class_centers)}."
            )
        class_features = _to_flat_numpy(class_features)

        class_name = class_id_to_name[class_id].split(',')[0].strip()
        for shift, center in enumerate(class_centers):
            source_paths, squared_distances = _nearest_image_paths(
                class_features, class_paths, center, args.cluster_caption_neighbor_count
            )
            missing = [path for path in source_paths if not os.path.isfile(path)]
            if missing:
                raise FileNotFoundError(f"Caption montage source image was not found: {missing[0]}")

            montage_path = os.path.join(args.cluster_caption_montage_dir, class_id, f"{shift}.png")
            _save_montage(source_paths, montage_path, args.cluster_caption_montage_tile_size)
            tasks.append((class_id, class_name, shift, montage_path))
            caption_inputs.setdefault(class_id, {})[str(shift)] = {
                "image_path": montage_path,
                "source_paths": source_paths,
                "squared_vae_distances": squared_distances,
            }

    return tasks, caption_inputs


def _build_caption_tasks(args, sel_classes, class_id_to_name):
    if args.cluster_caption_image_mode == "montage_neighbors":
        return _build_montage_tasks(args, sel_classes, class_id_to_name)

    tasks = [
        (
            class_id,
            class_id_to_name[class_id].split(',')[0].strip(),
            shift,
            os.path.join(args.save_dir, "real_images", class_id, f"{shift}.png"),
        )
        for class_id in sel_classes
        for shift in range(args.IPC)
    ]
    caption_inputs = {
        class_id: {
            str(shift): {"image_path": image_path, "source_paths": [image_path]}
            for task_class_id, _, shift, image_path in tasks
            if task_class_id == class_id
        }
        for class_id in sel_classes
    }
    return tasks, caption_inputs


def generate_cluster_captions(args, sel_classes, class_id_to_name):
    """Caption CoDA's saved representative images and write a reusable JSON manifest."""
    if not torch.cuda.is_available() and args.cluster_caption_device.startswith("cuda"):
        raise RuntimeError("--cluster_caption_device requests CUDA, but no CUDA device is available.")

    caption_path = args.cluster_caption_file
    if os.path.isfile(caption_path) and not args.overwrite_cluster_captions:
        _validate_caption_config(caption_path, args)
        captions = load_cluster_captions(caption_path, sel_classes, args.IPC)
        print(f"Using existing complete cluster captions: {caption_path}")
        return captions

    payload = _caption_file_payload(args)
    tasks, payload["caption_inputs"] = _build_caption_tasks(args, sel_classes, class_id_to_name)
    missing_images = [path for _, _, _, path in tasks if not os.path.isfile(path)]
    if missing_images:
        preview = ", ".join(missing_images[:5])
        suffix = " ..." if len(missing_images) > 5 else ""
        raise FileNotFoundError(
            f"Representative images are incomplete; missing {len(missing_images)} files: {preview}{suffix}. "
            "Run --calcu_cluster with the same experiment settings first."
        )

    device = args.cluster_caption_device
    print(f"Loading LLaVA caption model from: {args.cluster_caption_model_path}")
    processor, model, dtype = _load_llava(args.cluster_caption_model_path, device)

    for class_id, class_name, shift, image_path in tqdm(tasks, desc="Captioning cluster representatives"):
        instruction = args.cluster_caption_instruction.format(class_name=class_name)
        raw_caption = _generate_caption(
            model=model,
            processor=processor,
            dtype=dtype,
            device=device,
            image_path=image_path,
            instruction=instruction,
            max_new_tokens=args.cluster_caption_max_new_tokens,
        )
        caption = normalize_caption(
            raw_caption,
            image_mode=args.cluster_caption_image_mode,
            max_words=args.cluster_caption_max_words,
        )
        payload["raw_captions"].setdefault(class_id, {})[str(shift)] = raw_caption
        payload["captions"].setdefault(class_id, {})[str(shift)] = caption

    _write_json(caption_path, payload)
    print(f"Saved {len(tasks)} cluster captions to: {caption_path}")

    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return payload["captions"]
