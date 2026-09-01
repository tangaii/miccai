"""LoRA adapter training utilities for the shared multimodal backbone."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from medical_parsing.models.backbone import (
    apply_chat_template,
    build_prompt,
    configure_environment,
    messages_for,
)
from medical_parsing.schema import prepared_image, row_image_refs


def build_lora_model(
    base_path: str | Path,
    *,
    device_map: str | dict[str, Any] | None = None,
    rank: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    target_modules: Sequence[str] = ("q_proj", "k_proj", "v_proj", "o_proj"),
):
    """Load the external base and attach one standard LoRA adapter."""

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText

    configure_environment()
    model = AutoModelForImageTextToText.from_pretrained(
        str(base_path), torch_dtype=torch.bfloat16, device_map=device_map,
        trust_remote_code=True, local_files_only=True,
    )
    config = LoraConfig(
        r=rank, lora_alpha=alpha, lora_dropout=dropout,
        target_modules=list(target_modules), bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


class MultimodalSupervisionDataset:
    """Lazy records for adapter training.

    Each record must contain the unlabeled input fields plus a ``target``
    value supplied by the caller.  The target is appended through the same
    chat template used by inference.
    """

    def __init__(self, rows: list[dict[str, Any]], processor: Any, image_size: int = 896) -> None:
        self.rows = rows
        self.processor = processor
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        if "target" not in row:
            raise ValueError(f"adapter-training row has no target: {row.get('uid')}")
        text = messages_for(row, self.processor, answer=str(row["target"]))
        image = prepared_image(row_image_refs(row)[0], image_size=self.image_size)
        batch = self.processor(text=[text], images=[[image]], return_tensors="pt", padding=True)
        item = {key: value[0] if hasattr(value, "ndim") and value.ndim > 0 else value for key, value in batch.items()}
        labels = item["input_ids"].clone()
        if "attention_mask" in item:
            labels[item["attention_mask"] == 0] = -100
        item["labels"] = labels
        return item


def multimodal_collator(items: list[dict[str, Any]], pad_token_id: int = 0) -> dict[str, Any]:
    import torch
    from torch.nn.utils.rnn import pad_sequence

    keys = sorted(set().union(*(item.keys() for item in items)))
    result: dict[str, Any] = {}
    for key in keys:
        values = [item[key] for item in items if key in item]
        if len(values) != len(items):
            continue
        if values[0].ndim == 1 and len({int(value.shape[0]) for value in values}) > 1:
            padding_value = -100 if key == "labels" else 0 if key == "attention_mask" else pad_token_id
            result[key] = pad_sequence(values, batch_first=True, padding_value=padding_value)
        else:
            result[key] = torch.stack(values)
    return result


def train_lora_adapter(
    base_path: str | Path,
    processor: Any,
    rows: list[dict[str, Any]],
    output_dir: str | Path,
    *,
    epochs: int = 1,
    batch_size: int = 1,
    learning_rate: float = 2e-5,
    image_size: int = 896,
    device_map: str | dict[str, Any] | None = "auto",
) -> dict[str, Any]:
    """Train and save a task adapter; the base model remains external."""

    import torch
    from transformers import TrainingArguments, Trainer

    model = build_lora_model(base_path, device_map=device_map)
    dataset = MultimodalSupervisionDataset(rows, processor, image_size=image_size)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    arguments = TrainingArguments(
        output_dir=str(destination), num_train_epochs=epochs,
        per_device_train_batch_size=batch_size, learning_rate=learning_rate,
        logging_steps=1, save_strategy="no", report_to=[], remove_unused_columns=False,
        bf16=torch.cuda.is_available(),
    )
    collator = lambda items: multimodal_collator(items, pad_token_id=int(processor.tokenizer.pad_token_id or 0))
    trainer = Trainer(model=model, args=arguments, train_dataset=dataset, data_collator=collator)
    trainer.train()
    model.save_pretrained(str(destination), safe_serialization=True)
    processor.save_pretrained(str(destination))
    return {"status": "PASS", "output": str(destination), "rows": len(rows), "epochs": epochs}


__all__ = ["MultimodalSupervisionDataset", "build_lora_model", "multimodal_collator", "train_lora_adapter"]
