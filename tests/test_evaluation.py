from medical_parsing.evaluation.metrics import evaluate_rows


def test_evaluation_mixed_tasks():
    question = "Which? Options: A: normal B: abnormal"
    reference = [
        {"uid": "c", "task_type": "classification", "question": question, "answer": "A"},
        {"uid": "m", "task_type": "multi-label classification", "answer": ["dental caries"]},
        {"uid": "r", "task_type": "regression", "answer": 10},
    ]
    predictions = [
        {"uid": "c", "task_type": "classification", "prediction": "A"},
        {"uid": "m", "task_type": "multi-label classification", "prediction": "dental caries;"},
        {"uid": "r", "task_type": "regression", "prediction": "12"},
    ]
    result = evaluate_rows(reference, predictions)
    assert result["status"] == "PASS"
    assert result["tasks"]["classification"]["accuracy"] == 1.0
    assert result["tasks"]["regression"]["mae"] == 2.0
