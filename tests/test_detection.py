import json

import numpy as np
import pytest
from PIL import Image

from medical_parsing.models.detection_head import (
    FrozenSpatialQueryDecoder,
    load_frozen_spatial_query_decoder,
    make_frozen_spatial_query_decoder,
)
from medical_parsing.tasks.detection import decode_detection_outputs, extract_detection_features
from medical_parsing.training.detection import (
    detection_feature_uids,
    detection_loss,
    detection_uid_sha256,
    infer_query_count,
    matching,
    normalize_detection_uids,
    train_detection_head,
    validate_detection_feature_metadata,
)


def test_spatial_query_decoder_checkpoint_round_trip_and_shape_contract(tmp_path):
    import torch

    torch.manual_seed(3)
    source = make_frozen_spatial_query_decoder(1)
    checkpoint = tmp_path / "spatial_query_decoder.pt"
    torch.save({
        "state_dict": {key: value.detach().cpu() for key, value in source.state_dict().items()},
        "K": 1,
        "architecture": {
            "vision_dim": 2560, "hidden": 256, "heads": 8, "dropout": 0.1,
            "decoder_layers": 2, "box_mlp": "256->256->4 sigmoid", "presence": "256->1",
        },
    }, checkpoint)
    loaded, audit = load_frozen_spatial_query_decoder(checkpoint)
    assert isinstance(loaded, FrozenSpatialQueryDecoder)
    assert audit["K"] == 1
    assert audit["parameter_count"] == 3_485_445
    tokens = torch.zeros((2, 256, 2560))
    queries = torch.zeros((2, 2560))
    with torch.inference_mode():
        boxes, presence = loaded(tokens, queries)
    assert boxes.shape == (2, 1, 4)
    assert presence.shape == (2, 1)
    assert torch.all((boxes >= 0) & (boxes <= 1))


def test_detection_feature_extraction_uses_last_non_padding_state(tmp_path):
    import torch

    image_path = tmp_path / "image.png"
    Image.new("RGB", (12, 8), "white").save(image_path)
    rows = [
        {"uid": "a", "question": "return boxes", "images": [str(image_path)]},
        {"uid": "b", "question": "return boxes", "images": [str(image_path)]},
    ]

    class Processor:
        def apply_chat_template(self, messages, add_generation_prompt, tokenize):
            return messages[-1]["content"][-1]["text"]

        def __call__(self, text, images, return_tensors, padding):
            del images, return_tensors, padding
            return {
                "pixel_values": torch.zeros((len(text), 1, 3, 4, 4)),
                "input_ids": torch.ones((len(text), 6), dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]]),
            }

    class Model:
        def get_image_features(self, pixel_values, return_dict):
            del return_dict
            return type("Output", (), {
                "pooler_output": torch.zeros((pixel_values.shape[0], 256, 2560)),
            })()

        def __call__(self, **kwargs):
            pixels = kwargs["pixel_values"]
            if pixels.ndim == 5:
                raise RuntimeError("squeeze image axis")
            hidden = torch.zeros((pixels.shape[0], 6, 2560))
            hidden[0, 3] = 1.0
            hidden[1, 5] = 2.0
            return type("Output", (), {"hidden_states": (hidden,), "last_hidden_state": hidden})()

    tokens, queries = extract_detection_features(
        Model(), Processor(), rows, "cpu", type("Config", (), {"feature_batch_size": 2, "image_size": 896})(),
    )
    assert tokens.shape == (2, 256, 2560)
    assert queries.shape == (2, 2560)
    assert np.all(queries[0] == 1.0)
    assert np.all(queries[1] == 2.0)


def test_detection_decode_and_matching_contract():
    import torch

    serialized = decode_detection_outputs(
        torch.tensor([[[0.5, 0.5, 0.5, 0.5], [0.2, 0.2, 0.1, 0.1]]]),
        torch.tensor([[0.0, -10.0]]),
        [(100, 80)],
    )
    assert serialized == ["[[25.0,20.0,75.0,60.0]]"]
    matched, unmatched = matching(
        torch.tensor([[0.5, 0.5, 0.5, 0.5], [0.1, 0.1, 0.1, 0.1]]),
        torch.zeros(2),
        [[0.5, 0.5, 0.5, 0.5]],
    )
    assert matched == [(0, 0)]
    assert unmatched == [1]
    loss = detection_loss(
        torch.tensor([[[0.5, 0.5, 0.5, 0.5]]], requires_grad=True),
        torch.zeros((1, 1), requires_grad=True),
        [[[0.5, 0.5, 0.5, 0.5]]],
    )
    assert torch.isfinite(loss)


def test_detection_training_producer_is_replayable(tmp_path):
    tokens = np.zeros((2, 256, 2560), dtype=np.float32)
    queries = np.zeros((2, 2560), dtype=np.float32)
    targets = [
        [[0.5, 0.5, 0.4, 0.4]],
        [],
    ]
    first = tmp_path / "first.pt"
    second = tmp_path / "second.pt"
    result_first = train_detection_head(
        tokens, queries, targets, first, epochs=1, batch_size=2, device="cpu",
    )
    result_second = train_detection_head(
        tokens, queries, targets, second, epochs=1, batch_size=2, device="cpu",
    )
    assert result_first["K"] == 1
    assert result_first["parameter_count"] == 3_485_445
    assert [
        (item["loss"], item["lr"]) for item in result_first["loss_curve"]
    ] == [
        (item["loss"], item["lr"]) for item in result_second["loss_curve"]
    ]
    import torch
    first_state = torch.load(first, map_location="cpu", weights_only=False)["state_dict"]
    second_state = torch.load(second, map_location="cpu", weights_only=False)["state_dict"]
    assert first_state.keys() == second_state.keys()
    assert all(torch.equal(first_state[key], second_state[key]) for key in first_state)
    assert infer_query_count(targets) == 1


def test_detection_target_json_shape(tmp_path):
    from medical_parsing.training.detection import load_detection_targets

    path = tmp_path / "targets.json"
    path.write_text(json.dumps({"rows": [[[0.5, 0.5, 0.2, 0.3]], []]}), encoding="utf-8")
    assert load_detection_targets(path, expected_rows=2)[1] == []


def test_detection_target_and_feature_uid_alignment(tmp_path):
    from medical_parsing.training.detection import load_detection_targets

    path = tmp_path / "targets.json"
    path.write_text(json.dumps({
        "uids": ["a", "b"],
        "rows": [[[0.5, 0.5, 0.2, 0.3]], []],
    }), encoding="utf-8")
    arrays = {"uid": np.asarray(["a", "b"])}
    assert detection_feature_uids(arrays, 2) == ["a", "b"]
    assert normalize_detection_uids(np.asarray(["a", "b"]), 2) == ["a", "b"]
    assert load_detection_targets(
        path, expected_rows=2, expected_uids=detection_feature_uids(arrays, 2),
    )[0] == [[0.5, 0.5, 0.2, 0.3]]
    with pytest.raises(ValueError, match="UID order mismatch"):
        load_detection_targets(path, expected_rows=2, expected_uids=["b", "a"])


def test_detection_feature_metadata_is_a_reproducibility_lock():
    metadata = {
        "schema": "spatial_query_feature_cache_v1",
        "module": "Frozen Spatial Query Decoding",
        "image_token_shape": [2, 256, 2560],
        "query_state_shape": [2, 2560],
        "dtype": "float16",
        "image_size": 896,
        "feature_batch_size": 8,
    }
    uids = np.asarray(["a", "b"])
    metadata["uid_sha256"] = detection_uid_sha256(uids)
    arrays = {
        "metadata": np.asarray(json.dumps(metadata)),
        "uid": uids,
        "image_tokens": np.zeros((2, 256, 2560), dtype=np.float16),
        "query_states": np.zeros((2, 2560), dtype=np.float16),
    }
    assert validate_detection_feature_metadata(arrays, 2)["feature_batch_size"] == 8
    metadata["feature_batch_size"] = 2
    with pytest.raises(ValueError, match="metadata mismatch"):
        validate_detection_feature_metadata({**arrays, "metadata": np.asarray(json.dumps(metadata))}, 2)
