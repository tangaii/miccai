import pytest
from PIL import Image

from medical_parsing.models.backbone import (
    DecoderCapabilityError,
    LEGACY_VISION_ORIENTATION,
    WRAPPED_VISION_ORIENTATION,
    adapter_key_mapping,
    adapter_visual_orientation,
    extract_image_tokens,
)
from medical_parsing.config import ModelConfig


def _key(orientation: str) -> str:
    if orientation == LEGACY_VISION_ORIENTATION:
        return "base_model.model.model.vision_tower.encoder.layers.0.mlp.fc1.lora_A.weight"
    return "base_model.model.model.vision_tower.vision_model.encoder.layers.0.mlp.fc1.lora_A.weight"


@pytest.mark.parametrize(
    "source,target,should_map",
    [
        ("legacy", "legacy", False),
        ("legacy", "wrapped", True),
        ("wrapped", "legacy", True),
        ("wrapped", "wrapped", False),
    ],
)
def test_adapter_orientation_mapping_is_pairwise(source, target, should_map):
    assert adapter_visual_orientation([_key(source)]) == source
    assert adapter_visual_orientation([_key(target)]) == target
    mapping = adapter_key_mapping(source, target)
    assert bool(mapping) is should_map


def test_mixed_adapter_orientation_is_rejected():
    with pytest.raises(RuntimeError, match="mixed"):
        adapter_visual_orientation([_key("legacy"), _key("wrapped")])


def test_image_token_extraction_is_not_prompt_conditioned(tmp_path):
    import torch

    image_path = tmp_path / "image.png"
    Image.new("RGB", (8, 8), "white").save(image_path)

    class Processor:
        def apply_chat_template(self, messages, add_generation_prompt, tokenize):
            return messages[-1]["content"][-1]["text"]

        def __call__(self, text, images, return_tensors, padding):
            return {"pixel_values": torch.zeros((len(text), 1, 3, 4, 4))}

    class Model:
        def get_image_features(self, pixel_values, return_dict):
            return type("Output", (), {"pooler_output": torch.ones((pixel_values.shape[0], 256, 2560))})()

    rows = [{"uid": "x", "images": [str(image_path)]}]
    config = ModelConfig(feature_batch_size=1)
    first = extract_image_tokens(Model(), Processor(), rows, "cpu", config, prompt="first")
    second = extract_image_tokens(Model(), Processor(), rows, "cpu", config, prompt="second")
    assert first.shape == (1, 256, 2560)
    assert (first == second).all()


def test_adapter_capability_error_is_distinct_from_runtime_errors(monkeypatch):
    import medical_parsing.tasks.multilabel as multilabel
    import torch

    rows = [{"uid": "x"}]
    batch = {"input_ids": torch.zeros((1, 3), dtype=torch.long)}
    mapping = [(0, 0, [1])]
    monkeypatch.setattr(multilabel, "_teacher_forced_batch", lambda *args, **kwargs: (batch, mapping))
    monkeypatch.setattr(multilabel, "decoder_hidden_states", lambda *args, **kwargs: (_ for _ in ()).throw(DecoderCapabilityError("hidden states unavailable")))
    captured = {}

    def fake_full(logits, input_ids, tokenizer, mapping, rows, softcap=None):
        captured["softcap"] = softcap
        return torch.zeros((1, 1)).numpy()

    monkeypatch.setattr(multilabel, "score_logits_full", fake_full)
    monkeypatch.setattr(multilabel, "final_logit_softcap", lambda model: 7.0)

    class Model:
        def __call__(self, **kwargs):
            return type("Output", (), {"logits": torch.zeros((1, 3, 4))})()

    class Tokenizer:
        pass

    processor = type("Processor", (), {"tokenizer": Tokenizer()})()
    scores, implementation = multilabel.score_teacher_forced_streaming(
        Model(), processor, rows, "cpu", ["x"], ModelConfig(),
    )
    assert scores.shape == (1, 1)
    assert implementation.startswith("full-logit-compatibility-fallback:")
    assert captured["softcap"] == 7.0

    monkeypatch.setattr(multilabel, "decoder_hidden_states", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("scoring failure")))
    with pytest.raises(RuntimeError, match="scoring failure"):
        multilabel.score_teacher_forced_streaming(Model(), processor, rows, "cpu", ["x"], ModelConfig())
