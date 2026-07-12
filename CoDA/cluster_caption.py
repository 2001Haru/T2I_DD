"""Offline VLM captioning for CoDA cluster representative images."""

import json
import os
from datetime import datetime, timezone

import torch
from PIL import Image
from tqdm import tqdm


def _caption_file_payload(args):
    return {
        "metadata": {
            "format_version": 1,
            "model": args.cluster_caption_model_path,
            "instruction_template": args.cluster_caption_instruction,
            "max_new_tokens": args.cluster_caption_max_new_tokens,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        "captions": {},
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


def _validate_caption_config(path, args):
    with open(path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    metadata = payload.get("metadata", {})
    expected = {
        "model": args.cluster_caption_model_path,
        "instruction_template": args.cluster_caption_instruction,
        "max_new_tokens": args.cluster_caption_max_new_tokens,
    }
    mismatches = [
        key for key, value in expected.items() if metadata.get(key) != value
    ]
    if mismatches:
        raise ValueError(
            f"Caption configuration changed for {path}: {', '.join(mismatches)}. "
            "Use a new --cluster_caption_file or pass --overwrite_cluster_captions."
        )


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

    device = args.cluster_caption_device
    print(f"Loading LLaVA caption model from: {args.cluster_caption_model_path}")
    processor, model, dtype = _load_llava(args.cluster_caption_model_path, device)
    payload = _caption_file_payload(args)

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
    missing_images = [path for _, _, _, path in tasks if not os.path.isfile(path)]
    if missing_images:
        preview = ", ".join(missing_images[:5])
        suffix = " ..." if len(missing_images) > 5 else ""
        raise FileNotFoundError(
            f"Representative images are incomplete; missing {len(missing_images)} files: {preview}{suffix}. "
            "Run --calcu_cluster with the same experiment settings first."
        )

    for class_id, class_name, shift, image_path in tqdm(tasks, desc="Captioning cluster representatives"):
        instruction = args.cluster_caption_instruction.format(class_name=class_name)
        caption = _generate_caption(
            model=model,
            processor=processor,
            dtype=dtype,
            device=device,
            image_path=image_path,
            instruction=instruction,
            max_new_tokens=args.cluster_caption_max_new_tokens,
        )
        payload["captions"].setdefault(class_id, {})[str(shift)] = caption

    _write_json(caption_path, payload)
    print(f"Saved {len(tasks)} cluster captions to: {caption_path}")

    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return payload["captions"]
