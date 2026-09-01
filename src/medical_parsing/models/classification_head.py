"""Semantic image classification head and task ontology."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any


DENTAL_SLOTS = (
    "ROOT_FRAGMENT_MAXILLARY", "ROOT_FRAGMENT_MANDIBULAR",
    "TOOTH_LOSS_MAXILLARY", "TOOTH_LOSS_MANDIBULAR",
    "KENNEDY_MAXILLARY", "KENNEDY_MANDIBULAR",
)
SLOT_CLASSES = {
    "ROOT_FRAGMENT_MAXILLARY": ("root_no", "root_yes"),
    "ROOT_FRAGMENT_MANDIBULAR": ("root_no", "root_yes"),
    "TOOTH_LOSS_MAXILLARY": ("tooth_no_loss", "tooth_single_loss", "tooth_consecutive_multiple_loss", "tooth_edentulism"),
    "TOOTH_LOSS_MANDIBULAR": ("tooth_no_loss", "tooth_single_loss", "tooth_consecutive_multiple_loss", "tooth_edentulism"),
    "KENNEDY_MAXILLARY": ("kennedy_class_i", "kennedy_class_ii", "kennedy_class_iii", "kennedy_class_iv", "kennedy_edentulous", "kennedy_normal"),
    "KENNEDY_MANDIBULAR": ("kennedy_class_i", "kennedy_class_ii", "kennedy_class_iii", "kennedy_class_iv", "kennedy_edentulous", "kennedy_normal"),
    "fundus_diagnosis": ("normal", "diabetic retinopathy", "age-related macular degeneration", "glaucoma", "other"),
    "fundus_age": ("child", "young adult", "middle-aged", "elderly"),
    "bone_diagnosis": (
        "normal bone marrow", "myelofibrosis", "myelodysplastic syndrome", "multiple myeloma",
        "iron-deficiency anemia", "idiopathic thrombocytopenic purpura", "hemolytic anemia",
        "chronic myelogenous leukemia", "aplastic anemia", "acute myeloid leukemia", "acute lymphoblastic leukemia",
    ),
    "iugc_standard_plane": ("nonstandard_plane", "standard_plane"),
}
ROMAN_CLASS = re.compile(r"\bclass\s+(iv|iii|ii|i)\b")


def make_semantic_head():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class SemanticHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.token_ln = nn.LayerNorm(2560)
            self.token_projector = nn.Linear(2560, 128)
            self.bone_trunk = self._trunk(896)
            self.fundus_trunk = self._trunk(896)
            self.iugc_trunk = self._trunk(896)
            self.dental_trunk = self._trunk(1792)
            self.bone_head = nn.Linear(128, len(SLOT_CLASSES["bone_diagnosis"]))
            self.fundus_diagnosis_head = nn.Linear(128, len(SLOT_CLASSES["fundus_diagnosis"]))
            self.fundus_age_head = nn.Linear(128, len(SLOT_CLASSES["fundus_age"]))
            self.iugc_head = nn.Linear(128, len(SLOT_CLASSES["iugc_standard_plane"]))
            self.dental_root_head = nn.Linear(128, 2)
            self.dental_continuity_head = nn.Linear(128, 4)
            self.dental_kennedy_head = nn.Linear(128, 6)

        @staticmethod
        def _trunk(input_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.LayerNorm(input_dim), nn.Linear(input_dim, 256), nn.GELU(), nn.Dropout(.15),
                nn.Linear(256, 128), nn.GELU(),
            )

        @staticmethod
        def stats(tokens: Any) -> Any:
            quantiles = torch.quantile(tokens, torch.tensor([.10, .25, .50, .75, .90], device=tokens.device), dim=1)
            quantiles = quantiles.permute(1, 0, 2).reshape(tokens.shape[0], -1)
            return torch.cat([quantiles, tokens.mean(dim=1), tokens.std(dim=1, unbiased=False)], dim=1)

        def projected(self, tokens: Any) -> Any:
            return F.gelu(self.token_projector(self.token_ln(tokens.float())))

        def forward_source(self, tokens: Any, source: str) -> dict[str, Any]:
            projected = self.projected(tokens)
            global_stats = self.stats(projected)
            if source == "bone_marrow":
                return {"bone_diagnosis": self.bone_head(self.bone_trunk(global_stats))}
            if source == "fundus":
                representation = self.fundus_trunk(global_stats)
                return {
                    "fundus_diagnosis": self.fundus_diagnosis_head(representation),
                    "fundus_age": self.fundus_age_head(representation),
                }
            if source == "iugc":
                return {"iugc_standard_plane": self.iugc_head(self.iugc_trunk(global_stats))}
            if source != "dental":
                raise ValueError(source)
            grid = projected.reshape(projected.shape[0], 16, 16, 128)
            upper = self.stats(grid[:, :8].reshape(grid.shape[0], -1, 128))
            lower = self.stats(grid[:, 8:].reshape(grid.shape[0], -1, 128))
            upper_rep = self.dental_trunk(torch.cat([global_stats, upper], dim=1))
            lower_rep = self.dental_trunk(torch.cat([global_stats, lower], dim=1))
            return {
                "ROOT_FRAGMENT_MAXILLARY": self.dental_root_head(upper_rep),
                "ROOT_FRAGMENT_MANDIBULAR": self.dental_root_head(lower_rep),
                "TOOTH_LOSS_MAXILLARY": self.dental_continuity_head(upper_rep),
                "TOOTH_LOSS_MANDIBULAR": self.dental_continuity_head(lower_rep),
                "KENNEDY_MAXILLARY": self.dental_kennedy_head(upper_rep),
                "KENNEDY_MANDIBULAR": self.dental_kennedy_head(lower_rep),
            }

    return SemanticHead()


def load_semantic_heads(asset_dir: Path, device: str) -> list[Any]:
    import torch

    path = asset_dir / "classification_heads.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    folds = payload.get("folds", {})
    if set(folds) != {"0", "1", "2"}:
        raise RuntimeError("classification head pack must contain three folds")
    models = []
    for fold in ("0", "1", "2"):
        model = make_semantic_head().to(device)
        model.load_state_dict(folds[fold]["state_dict"], strict=True)
        model.eval()
        models.append(model)
    return models


def resolve_slot(row: dict[str, Any], parse_choices) -> str:
    source = str(row["dataset"])
    question = str(row.get("source_question") or row.get("question") or row.get("prompt")).lower()
    choices = parse_choices(str(row.get("source_question") or row.get("question") or row.get("prompt")))
    ontology = {text.lower() for _, text in choices}
    if source == "dental":
        jaw = "MAXILLARY" if re.search(r"\b(maxillary|maxilla|upper)\b", question) else "MANDIBULAR" if re.search(r"\b(mandibular|mandible|lower)\b", question) else None
        kind = "KENNEDY" if "kennedy" in question or "partial edentulism" in question else "ROOT_FRAGMENT" if "root" in question else "TOOTH_LOSS" if "continuity" in question or "tooth loss" in question else None
        if jaw and kind:
            return f"{kind}_{jaw}"
    if source == "fundus":
        if ontology == {"normal", "diabetic retinopathy", "age-related macular degeneration", "glaucoma", "other"} or any(word in question for word in ("diagnosis", "pathology", "retinal condition")):
            return "fundus_diagnosis"
        if any(word in question for word in ("age group", "age category", "age range")):
            return "fundus_age"
    if source == "bone_marrow":
        return "bone_diagnosis"
    if source == "iugc":
        return "iugc_standard_plane"
    raise RuntimeError(f"unable to resolve semantic classification slot for {row['uid']}")


def semantic_concept(source: str, slot: str, text: str, normalize) -> str | None:
    value = normalize(text)
    if source == "dental":
        if slot.startswith("ROOT_FRAGMENT"):
            return {"no": "root_no", "yes": "root_yes"}.get(value)
        if slot.startswith("TOOTH_LOSS"):
            if "no loss" in value:
                return "tooth_no_loss"
            if "single tooth loss" in value:
                return "tooth_single_loss"
            if "consecutive multiple" in value:
                return "tooth_consecutive_multiple_loss"
            if "edentul" in value:
                return "tooth_edentulism"
        if slot.startswith("KENNEDY"):
            match = ROMAN_CLASS.search(value)
            if match:
                return f"kennedy_class_{match.group(1)}"
            if "edentulous" in value or "edentulism" in value:
                return "kennedy_edentulous"
            if "normal" in value and "all teeth present" in value:
                return "kennedy_normal"
    if source == "fundus":
        if slot == "fundus_age":
            for concept in ("child", "young adult", "middle-aged", "elderly"):
                if value.startswith(concept):
                    return concept
        if slot == "fundus_diagnosis":
            for concept in ("age-related macular degeneration", "diabetic retinopathy", "normal", "glaucoma", "other"):
                if value.startswith(concept):
                    return concept
    if source == "bone_marrow" and value in SLOT_CLASSES["bone_diagnosis"]:
        return value
    if source == "iugc":
        return {"standard plane": "standard_plane", "non-standard plane": "nonstandard_plane"}.get(value)
    return None


def serialize_concept(row: dict[str, Any], slot: str, concept: str, parse_choices, normalize) -> str:
    choices = parse_choices(str(row.get("source_question") or row.get("question") or row.get("prompt")))
    matches = [letter for letter, text in choices if semantic_concept(str(row["dataset"]), slot, text, normalize) == concept]
    if len(matches) != 1:
        raise RuntimeError(f"semantic mapper has {len(matches)} matches for {row['uid']} / {slot} / {concept}")
    return matches[0]

