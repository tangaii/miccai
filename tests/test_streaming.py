import numpy as np

from medical_parsing.tasks.multilabel import score_hidden_streaming, score_logits_full


class DummyTokenizer:
    def encode(self, value, add_special_tokens=False):
        if value == "<start_of_turn>model\n":
            return [90, 91]
        if value == "model\n":
            return [90, 91]
        return {"a": [5, 6], "b": [7]}[value]


def test_token_chunk_streaming_matches_full_reference():
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
    streamed = score_hidden_streaming(hidden, head, input_ids, tokenizer, mapping, rows, token_chunk=16, softcap=2.0)
    np.testing.assert_allclose(streamed, full, rtol=1e-5, atol=1e-5)
