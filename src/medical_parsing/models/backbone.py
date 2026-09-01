"""MedGemma loading, prompt rendering, and visual feature extraction."""

from __future__ import annotations

import gc
import io
import os
from pathlib import Path
import random
from typing import Any

import numpy as np

from medical_parsing.config import ModelConfig
from medical_parsing.schema import load_image, prepared_image, row_image_refs


EXPECTED_BASE_MODEL = "google/medgemma-1.5-4b-it"
SYSTEM_PROMPT = "You are an expert medical imaging assistant."
TASK_INSTRUCTIONS = {
    "classification": (
        "You are given a medical image. Answer the classification question using only the provided image. "
        "If options are provided, return only the correct option letter or class label."
    ),
    "multi-label classification": (
        "You are given a medical image. Identify all applicable findings or labels. "
        "Return labels separated by semicolons. Return an empty string if none apply."
    ),
    "regression": "You are given a medical image. Return only the requested numeric measurement.",
}


def configure_environment() -> None:
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def require_cuda(device: str) -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for multimodal inference")
    torch.cuda.set_device(torch.device(device))


def set_determinism(seed: int = 0) -> dict[str, Any]:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return {
        "seed": seed,
        "torch_deterministic_algorithms": True,
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
    }


def apply_chat_template(processor: Any, messages: list[dict[str, Any]], generation: bool) -> str:
    if hasattr(processor, "apply_chat_template"):
        rendered = processor.apply_chat_template(messages, add_generation_prompt=generation, tokenize=False)
    else:
        rendered = processor.tokenizer.apply_chat_template(messages, add_generation_prompt=generation, tokenize=False)
    return rendered if generation else rendered.strip()


def build_prompt(row: dict[str, Any]) -> str:
    task = row["task_type"]
    prompt = str(row.get("prompt") or row.get("question") or "")
    choices = row.get("choices")
    if isinstance(choices, str):
        try:
            import json
            choices = json.loads(choices)
        except json.JSONDecodeError:
            choices = None
    if not isinstance(choices, list):
        choices = []
    parts = [TASK_INSTRUCTIONS[task]]
    if prompt:
        parts.append(prompt)
    if choices and "options:" not in prompt.lower():
        parts.append("Options: " + "; ".join(str(choice) for choice in choices))
    return "\n\n".join(parts)


def messages_for(row: dict[str, Any], processor: Any, answer: str | None = None) -> str:
    content = [{"type": "image"}, {"type": "text", "text": build_prompt(row)}]
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": content},
    ]
    if answer is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return apply_chat_template(processor, messages, generation=answer is None)


def processor_batch(processor: Any, row: dict[str, Any], text: str, image_size: int) -> dict[str, Any]:
    image = prepared_image(row_image_refs(row)[0], image_size=image_size)
    return processor(text=[text], images=[[image]], return_tensors="pt", padding=True)


def load_adapter_bundle(
    base_path: Path,
    adapter_path: Path,
    device: str,
    model_config: ModelConfig,
) -> tuple[Any, Any, dict[str, Any]]:
    import json
    import torch
    from peft import PeftModel
    from safetensors import safe_open
    from transformers import AutoModelForImageTextToText, AutoProcessor

    require_cuda(device)
    if not base_path.is_dir() or not adapter_path.is_dir():
        raise FileNotFoundError(f"base or adapter missing: {base_path} / {adapter_path}")
    adapter_config_path = adapter_path / "adapter_config.json"
    if not adapter_config_path.is_file():
        raise FileNotFoundError(adapter_config_path)
    adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
    if adapter_config.get("base_model_name_or_path") != model_config.name:
        raise RuntimeError(f"adapter base identity mismatch: {adapter_config.get('base_model_name_or_path')!r}")
    if adapter_config.get("peft_type") != "LORA":
        raise RuntimeError(f"unsupported adapter type: {adapter_config.get('peft_type')!r}")
    processor = AutoProcessor.from_pretrained(str(adapter_path), trust_remote_code=True, local_files_only=True, use_fast=True)
    processor.tokenizer.padding_side = "right"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    model = AutoModelForImageTextToText.from_pretrained(
        str(base_path), torch_dtype=torch.bfloat16, device_map={"": device},
        trust_remote_code=True, local_files_only=True,
    )
    adapter_weights = adapter_path / "adapter_model.safetensors"
    adapter_keys: list[str] = []
    if adapter_weights.is_file():
        with safe_open(str(adapter_weights), framework="pt", device="cpu") as handle:
            adapter_keys = list(handle.keys())
    legacy_vision_path = any("vision_tower.encoder." in key for key in adapter_keys)
    key_mapping = {r"^model\.vision_tower\.encoder": "model.vision_tower.vision_model.encoder"} if legacy_vision_path else None
    model = PeftModel.from_pretrained(
        model, str(adapter_path), is_trainable=False, local_files_only=True, key_mapping=key_mapping,
    )
    if set(model.peft_config) != {"default"}:
        raise RuntimeError(f"unexpected adapter stack: {sorted(model.peft_config)}")
    model.eval()
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.pad_token_id = processor.tokenizer.pad_token_id
    return model, processor, {
        "kind": "single_lora_adapter",
        "base_model_name_or_path": adapter_config["base_model_name_or_path"],
        "adapter_path": str(adapter_path),
        "legacy_vision_path_remapped": legacy_vision_path,
    }


def load_raw_bundle(base_path: Path, device: str, model_config: ModelConfig) -> tuple[Any, Any, dict[str, Any]]:
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    require_cuda(device)
    if not base_path.is_dir():
        raise FileNotFoundError(base_path)
    processor = AutoProcessor.from_pretrained(str(base_path), trust_remote_code=True, local_files_only=True, use_fast=True)
    processor.tokenizer.padding_side = "right"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    model = AutoModelForImageTextToText.from_pretrained(
        str(base_path), torch_dtype=torch.bfloat16, device_map={"": device},
        trust_remote_code=True, local_files_only=True,
    )
    model.eval()
    return model, processor, {"kind": "raw_base"}


def image_features(
    model: Any,
    processor: Any,
    rows: list[dict[str, Any]],
    device: str,
    model_config: ModelConfig,
    prompt: str = "image feature",
) -> np.ndarray:
    import torch

    parts: list[np.ndarray] = []
    for start in range(0, len(rows), model_config.feature_batch_size):
        subset = rows[start:start + model_config.feature_batch_size]
        images = [prepared_image(row_image_refs(row)[0], model_config.image_size) for row in subset]
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        rendered = [apply_chat_template(processor, messages, generation=True) for _ in images]
        batch = processor(text=rendered, images=[[image] for image in images], return_tensors="pt", padding=True)
        pixels = batch["pixel_values"]
        if pixels.ndim == 5:
            pixels = pixels[:, 0]
        with torch.inference_mode():
            output = model.get_image_features(pixel_values=pixels.to(device), return_dict=True).pooler_output
        result = output.float().cpu().numpy().astype(np.float16)
        parts.append(result)
        del batch, pixels, output, images, rendered
        torch.cuda.empty_cache()
    result = np.concatenate(parts, axis=0)
    if result.shape != (len(rows), 256, 2560) or not np.isfinite(result).all():
        raise RuntimeError(f"unexpected image-token shape {result.shape}")
    return result


def generated_text(model: Any, processor: Any, row: dict[str, Any], device: str, model_config: ModelConfig) -> str:
    import torch

    text = messages_for(row, processor)
    batch = processor_batch(processor, row, text, model_config.image_size)
    batch = {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}
    with torch.inference_mode():
        output = model.generate(
            **batch, max_new_tokens=model_config.max_new_tokens, do_sample=False, top_p=1.0,
            pad_token_id=processor.tokenizer.pad_token_id,
        )
    input_len = int(batch["input_ids"].shape[1])
    return processor.tokenizer.decode(output[0, input_len:], skip_special_tokens=True).strip()


def prompt_only_generated_text(model: Any, processor: Any, row: dict[str, Any], device: str, model_config: ModelConfig) -> str:
    import torch

    content = [{"type": "image"}, {"type": "text", "text": str(row["prompt"])}]
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": content},
    ]
    text = apply_chat_template(processor, messages, generation=True)
    batch = processor(text=[text], images=[[prepared_image(row_image_refs(row)[0], model_config.image_size)]], return_tensors="pt", padding=True)
    batch = {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}
    with torch.inference_mode():
        output = model.generate(
            **batch, max_new_tokens=model_config.max_new_tokens, do_sample=False, top_p=1.0,
            pad_token_id=processor.tokenizer.pad_token_id,
        )
    input_len = int(batch["input_ids"].shape[1])
    return processor.tokenizer.decode(output[0, input_len:], skip_special_tokens=True).strip()


def unwrap_base_model(model: Any) -> Any:
    if hasattr(model, "get_base_model"):
        base = model.get_base_model()
        if base is not None:
            return base
    return model


def decoder_hidden_states(model: Any, batch: dict[str, Any]) -> tuple[Any, Any, float | None]:
    """Run the decoder without constructing the full vocabulary logits tensor."""

    base = unwrap_base_model(model)
    lm_head = getattr(base, "lm_head", None)
    decoder = getattr(base, "model", None)
    if lm_head is None or decoder is None:
        raise TypeError("model does not expose a decoder and language-model head")
    allowed = {"input_ids", "pixel_values", "attention_mask", "position_ids", "token_type_ids", "cache_position", "inputs_embeds"}
    decoder_batch = {key: value for key, value in batch.items() if key in allowed}
    outputs = decoder(**decoder_batch, use_cache=False, return_dict=True)
    config = getattr(base, "config", None)
    text_config = getattr(config, "text_config", config)
    softcap = getattr(text_config, "final_logit_softcapping", None)
    return outputs.last_hidden_state, lm_head, softcap


def clear_model(model: Any, processor: Any) -> None:
    del model, processor
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

