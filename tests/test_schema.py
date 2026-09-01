import json

import pytest
from PIL import Image

from medical_parsing.schema import (
    TASK_CLASSIFICATION,
    atomic_write_jsonl,
    parse_choices,
    prepared_image,
    read_records,
    validate_input_rows,
    validate_output_rows,
)
from medical_parsing.tasks.multilabel import ATOMS, serialize_effective


def test_input_and_output_schema(tmp_path):
    image_path = tmp_path / "image.png"
    Image.new("RGB", (20, 10), "white").save(image_path)
    question = "Which finding? Options: A: normal B: abnormal"
    input_path = tmp_path / "input.jsonl"
    atomic_write_jsonl(input_path, [{
        "uid": "x", "task_type": "classification", "dataset": "demo",
        "prompt": question, "question": question, "images": [str(image_path)],
    }])
    rows = validate_input_rows(read_records(input_path), image_root=tmp_path)
    assert rows[0]["task_type"] == TASK_CLASSIFICATION
    assert parse_choices(question) == [("A", "normal"), ("B", "abnormal")]
    assert prepared_image(str(image_path)).size == (896, 896)
    output = [{"uid": "x", "task_type": "classification", "prediction": "A"}]
    assert validate_output_rows(rows, output, set()) == {"rows": 1, "status": "PASS"}


def test_comma_inside_a_canonical_atom_is_preserved():
    assert validate_output_rows(
        [{"uid": "m", "task_type": "multi-label classification", "source_question": "report", "images": ["unused"], "dataset": "demo"}],
        [{"uid": "m", "task_type": "multi-label classification", "prediction": serialize_effective({ATOMS[5]})}],
        set(ATOMS),
    )["status"] == "PASS"


def test_labeled_input_is_rejected(tmp_path):
    image_path = tmp_path / "image.png"
    Image.new("RGB", (5, 5), "white").save(image_path)
    with pytest.raises(ValueError, match="labeled input"):
        validate_input_rows([{
            "uid": "x", "task_type": "regression", "dataset": "demo",
            "prompt": "measure", "images": [str(image_path)], "answer": 3,
        }], image_root=tmp_path)
