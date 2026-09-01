#!/usr/bin/env python3
"""Create a tiny legal synthetic fixture for contract and CLI smoke tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from medical_parsing.schema import atomic_write_jsonl  # noqa: E402


MLC_PROMPT = (
    "As a dental imaging specialist, what abnormalities would you report on tooth 11? "
    "Select all that apply from the following, separate multiple selections with a semicolon: "
    "dental caries; periradicular lesions; pulp diseases; chronic injury of tooth hard tissues; "
    "disturbances of eruption of teeth; periodontal diseases; endodontic treatment, restorative treatment, and complications."
)


def _image(path: Path, index: int) -> None:
    image = Image.new("RGB", (192, 128), (245 - index * 20, 245, 245))
    draw = ImageDraw.Draw(image)
    draw.rectangle((20 + index * 12, 20, 160, 105), outline=(30, 70, 120), width=4)
    draw.ellipse((55, 35, 125, 95), outline=(160, 50, 50), width=3)
    image.save(path, format="PNG")


def main() -> int:
    parser = argparse.ArgumentParser(description="Create synthetic public smoke data outside the repository.")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    choices = [
        "A: normal bone marrow", "B: myelofibrosis", "C: myelodysplastic syndrome",
        "D: multiple myeloma", "E: iron-deficiency anemia", "F: idiopathic thrombocytopenic purpura",
        "G: hemolytic anemia", "H: chronic myelogenous leukemia", "I: aplastic anemia",
        "J: acute myeloid leukemia", "K: acute lymphoblastic leukemia",
    ]
    specs = [
        ("smoke-cls", "bone_marrow", "Question: based on the image, which condition is shown? Options: " + " ".join(choices)),
        ("smoke-mlc", "dental", MLC_PROMPT),
        ("smoke-reg", "measurement", "Return the requested numeric measurement in the image."),
    ]
    for index, (uid, dataset, prompt) in enumerate(specs):
        image_path = args.output_dir / f"{uid}.png"
        _image(image_path, index)
        rows.append({
            "uid": uid, "task_type": {"smoke-cls": "classification", "smoke-mlc": "multi-label classification", "smoke-reg": "regression"}[uid],
            "dataset": dataset, "prompt": prompt, "question": prompt, "images": [str(image_path)],
        })
    input_path = args.output_dir / "input.jsonl"
    atomic_write_jsonl(input_path, rows)
    print(json.dumps({"status": "PASS", "input": str(input_path), "rows": len(rows)}, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
