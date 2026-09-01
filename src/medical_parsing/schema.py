"""Input, image, and prediction serialization contracts."""

from __future__ import annotations

import hashlib
import io
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable
import unicodedata
import zipfile

from PIL import Image, ImageOps


TASK_CLASSIFICATION = "classification"
TASK_MULTILABEL = "multi-label classification"
TASK_REGRESSION = "regression"
SUPPORTED_TASKS = {TASK_CLASSIFICATION, TASK_MULTILABEL, TASK_REGRESSION}
LABEL_KEYS = {
    "answer", "raw_answer", "target", "reference", "label", "ground_truth", "gold",
    "gold_answer", "target_value", "raw_target", "prediction", "pred", "output",
}
CHOICE_MARKER = re.compile(r"(?<![A-Za-z])([A-K])\s*[:.)]\s*", re.IGNORECASE)


def maybe_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text or text[0] not in "[{\"":
        return value
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).lower()
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    return re.sub(r"\s+", " ", text).strip()


def canonical_task(value: Any) -> str:
    key = str(value or "").strip().lower().replace("_", " ")
    if key in {"classification", "disease diagnosis classification", "cls"}:
        return TASK_CLASSIFICATION
    if key in {"multi label classification", "multi-label classification", "multilabel", "multi label", "multi_label"}:
        return TASK_MULTILABEL
    if key == TASK_REGRESSION:
        return TASK_REGRESSION
    raise ValueError(f"unsupported task_type={value!r}")


def read_records(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.suffix.lower() == ".json":
        value = json.loads(source.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            value = value.get("rows", value.get("data", value))
        if not isinstance(value, list):
            raise ValueError("JSON input must be a list of objects")
        rows = value
    else:
        rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("every input record must be a JSON object")
    return [dict(row) for row in rows]


def row_image_refs(row: dict[str, Any]) -> list[str]:
    value = row.get("images") or row.get("image_paths") or row.get("image") or row.get("image_path")
    value = maybe_json(value)
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"missing image for uid={row.get('uid')}")
    refs = [str(item).strip() for item in value if str(item).strip()]
    if not refs:
        raise ValueError(f"missing image for uid={row.get('uid')}")
    return refs


def normalize_image_ref(value: str, image_root: Path | None = None, repo_root: Path | None = None) -> str:
    value = str(value).strip()
    from urllib.parse import urlsplit
    if urlsplit(value).scheme in {"http", "https"}:
        raise ValueError("remote image references are not allowed; use local files")
    if value.startswith("zip://"):
        archive_name, member = value[len("zip://"):].split("::", 1)
        archive_path = Path(archive_name)
        if not archive_path.is_absolute():
            candidates = [root / archive_path for root in (image_root, repo_root) if root is not None]
            archive_path = next((item for item in candidates if item.is_file()), candidates[0] if candidates else archive_path)
        if not archive_path.is_file():
            raise FileNotFoundError(archive_path)
        with zipfile.ZipFile(archive_path) as archive:
            if member not in archive.namelist():
                raise FileNotFoundError(f"archive member does not exist: {member}")
        return f"zip://{archive_path}::{member}"
    path = Path(value[7:] if value.startswith("file://") else value)
    if not path.is_absolute():
        candidates = [root / path for root in (image_root, repo_root) if root is not None]
        path = next((item for item in candidates if item.is_file()), candidates[0] if candidates else path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return str(path)


def parse_choices(question: str) -> list[tuple[str, str]]:
    markers = list(CHOICE_MARKER.finditer(str(question)))
    labels = [marker.group(1).upper() for marker in markers]
    expected = [chr(ord("A") + index) for index in range(len(labels))]
    if len(markers) < 2 or labels != expected:
        raise ValueError(f"classification options must be consecutive A..K: {labels!r}")
    values: list[tuple[str, str]] = []
    for index, marker in enumerate(markers):
        stop = markers[index + 1].start() if index + 1 < len(markers) else len(question)
        values.append((marker.group(1).upper(), normalize_text(question[marker.end():stop]).strip(" ,;")))
    return values


def validate_input_rows(raw_rows: list[dict[str, Any]], image_root: Path | None = None, repo_root: Path | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in raw_rows:
        forbidden = sorted(key for key in raw if str(key).lower() in LABEL_KEYS)
        if forbidden:
            raise ValueError(f"labeled input is forbidden; uid={raw.get('uid')} keys={forbidden}")
        uid = str(raw.get("uid", "")).strip()
        if not uid or uid in seen:
            raise ValueError(f"uid missing or duplicated: {uid!r}")
        task = canonical_task(raw.get("task_type", raw.get("task")))
        prompt = str(raw.get("prompt") or raw.get("question") or "").strip()
        if not prompt:
            raise ValueError(f"prompt/question missing for uid={uid}")
        refs = [normalize_image_ref(value, image_root=image_root, repo_root=repo_root) for value in row_image_refs(raw)]
        dataset = str(raw.get("dataset") or raw.get("source") or "").strip()
        if not dataset:
            raise ValueError(f"dataset/source missing for uid={uid}")
        row = dict(raw)
        row.update({
            "uid": uid,
            "task_type": task,
            "images": refs,
            "prompt": prompt,
            "question": str(raw.get("question") or prompt),
            "source_question": str(raw.get("source_question") or raw.get("question") or prompt),
            "dataset": dataset,
        })
        choices = maybe_json(raw.get("choices"))
        if choices is not None and not isinstance(choices, list):
            raise ValueError(f"choices must be a list for uid={uid}")
        if task == TASK_CLASSIFICATION:
            markers = list(CHOICE_MARKER.finditer(row["source_question"]))
            if markers:
                labels = [marker.group(1).upper() for marker in markers]
                expected = [chr(ord("A") + index) for index in range(len(labels))]
                if len(markers) < 2 or labels != expected:
                    raise ValueError(f"classification options must be consecutive A..K for uid={uid}")
        seen.add(uid)
        rows.append(row)
    if not rows:
        raise ValueError("input contains no rows")
    return rows


def uri_bytes(uri: str) -> bytes:
    value = str(uri)
    if value.startswith("zip://"):
        archive_name, member = value[len("zip://"):].split("::", 1)
        with zipfile.ZipFile(Path(archive_name)) as archive:
            return archive.read(member)
    path = Path(value[7:] if value.startswith("file://") else value)
    return path.read_bytes()


def load_image(uri: str) -> Image.Image:
    with Image.open(io.BytesIO(uri_bytes(uri))) as opened:
        image = opened.copy()
    return ImageOps.exif_transpose(image).convert("RGB")


def image_sha(uri: str) -> str:
    return hashlib.sha256(uri_bytes(uri)).hexdigest()


def prepared_image(uri: str, image_size: int = 896) -> Image.Image:
    return load_image(uri).resize((image_size, image_size), Image.Resampling.BICUBIC)


def atomic_write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(target)


def validate_output_rows(input_rows: list[dict[str, Any]], output_rows: list[dict[str, Any]], atoms: set[str]) -> dict[str, Any]:
    if len(input_rows) != len(output_rows):
        raise ValueError(f"output row count mismatch: {len(output_rows)} != {len(input_rows)}")
    expected_uids = [row["uid"] for row in input_rows]
    actual_uids = [str(row.get("uid", "")) for row in output_rows]
    if actual_uids != expected_uids:
        raise ValueError("output UID order/coverage mismatch")
    for source, record in zip(input_rows, output_rows):
        if set(record) != {"uid", "task_type", "prediction"}:
            raise ValueError(f"non-canonical output keys for {record.get('uid')}: {sorted(record)}")
        if canonical_task(record["task_type"]) != source["task_type"]:
            raise ValueError(f"task mismatch for {record['uid']}")
        prediction = record["prediction"]
        if source["task_type"] == TASK_CLASSIFICATION:
            if str(prediction).upper() not in {letter for letter, _ in parse_choices(source["source_question"])}:
                raise ValueError(f"illegal classification prediction for {record['uid']}: {prediction!r}")
        elif source["task_type"] == TASK_MULTILABEL:
            labels = parse_label_set(prediction)
            if not labels.issubset(atoms):
                raise ValueError(f"illegal multi-label atoms for {record['uid']}: {sorted(labels - atoms)}")
        else:
            value = float(prediction)
            if not math.isfinite(value) or not 0.0 <= value <= 100.0:
                raise ValueError(f"illegal regression prediction for {record['uid']}: {prediction!r}")
    return {"rows": len(output_rows), "status": "PASS"}


def parse_label_set(value: Any) -> set[str]:
    value = maybe_json(value)
    if value is None:
        return set()
    if isinstance(value, dict):
        if "labels" in value:
            return parse_label_set(value["labels"])
        values = [key for key, flag in value.items() if bool(flag)]
    elif isinstance(value, (list, tuple, set)):
        values = list(value)
    else:
        text = str(value).strip()
        if normalize_text(text) in {"", "none", "no finding", "no findings", "n/a", "na", "null", "[]"}:
            return set()
        # Canonical multi-label output uses semicolons.  A valid atom itself
        # contains commas, so commas are separators only when no stronger
        # canonical delimiter is present.
        if any(mark in text for mark in ";\n|"):
            values = re.split(r"[;\n|]+", text)
        elif "," in text:
            values = text.split(",")
        else:
            values = [text]
    return {normalize_text(item) for item in values if normalize_text(item) not in {"", "none", "no finding", "no findings", "n/a", "na", "null", "[]"}}
