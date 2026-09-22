import numpy as np
import pytest

from medical_parsing.tasks.multilabel import score_hidden_streaming, score_logits_full


class DummyTokenizer:
    def encode(self, value, add_special_tokens=False):
        if value == "<start_of_turn>model\n":
            return [90, 91]
        if value == "model\n":
            return [90, 91]
        return {"a": [5, 6], "b": [7]}[value]


@pytest.mark.parametrize("chunk_size", [1, 5, 16, 64])
def test_vocab_chunk_streaming_matches_full_reference(chunk_size):
    import torch

    torch.manual_seed(3)
    tokenizer = DummyTokenizer()
    rows = [{"uid": "r0"}, {"uid": "r1"}]
    mapping = [
        (0, 0, [5, 6]), (0, 1, [7]),
        (1, 0, [5, 6]), (1, 1, [7]),
    ]
    input_ids = torch.tensor([
        [1, 90, 91, 3, 5, 6, 0, 0],
        [1, 90, 91, 3, 7, 0, 0, 0],
        [2, 90, 91, 4, 5, 6, 0, 0],
        [2, 90, 91, 4, 7, 0, 0, 0],
    ])
    hidden = torch.randn(4, 8, 11)
    head = torch.nn.Linear(11, 37, bias=True)
    logits = head(hidden)
    full = score_logits_full(logits, input_ids, tokenizer, mapping, rows, softcap=2.0)
    streamed = score_hidden_streaming(hidden, head, input_ids, tokenizer, mapping, rows, vocab_chunk_size=chunk_size, softcap=2.0)
    np.testing.assert_allclose(streamed, full, rtol=1e-5, atol=1e-5)


def test_legacy_token_chunk_keyword_keeps_the_same_math():
    import torch

    torch.manual_seed(3)
    tokenizer = DummyTokenizer()
    rows = [{"uid": "r0"}, {"uid": "r1"}]
    mapping = [(0, 0, [5, 6]), (1, 0, [7])]
    input_ids = torch.tensor([
        [1, 90, 91, 3, 5, 6, 0],
        [2, 90, 91, 4, 7, 0, 0],
    ])
    hidden = torch.randn(2, 7, 11)
    head = torch.nn.Linear(11, 37, bias=True)
    expected = score_hidden_streaming(hidden, head, input_ids, tokenizer, mapping, rows, vocab_chunk_size=16)
    actual = score_hidden_streaming(hidden, head, input_ids, tokenizer, mapping, rows, token_chunk=16)
    np.testing.assert_array_equal(actual, expected)


def test_bfloat16_hidden_state_streaming_matches_full_when_supported():
    import torch

    try:
        torch.manual_seed(4)
        tokenizer = DummyTokenizer()
        rows = [{"uid": "r0"}, {"uid": "r1"}]
        mapping = [(0, 0, [5, 6]), (1, 0, [7])]
        input_ids = torch.tensor([
            [1, 90, 91, 3, 5, 6, 0],
            [2, 90, 91, 4, 7, 0, 0],
        ])
        hidden = torch.randn(2, 7, 11).to(torch.bfloat16)
        head = torch.nn.Linear(11, 37, bias=True).to(torch.bfloat16)
        logits = head(hidden)
    except RuntimeError as exc:  # pragma: no cover - hardware/backend dependent
        pytest.skip(f"bfloat16 linear layer unavailable: {exc}")
    full = score_logits_full(logits, input_ids, tokenizer, mapping, rows, softcap=2.0)
    streamed = score_hidden_streaming(hidden, head, input_ids, tokenizer, mapping, rows, vocab_chunk_size=7, softcap=2.0)
    np.testing.assert_allclose(streamed, full, rtol=3e-2, atol=3e-2)
