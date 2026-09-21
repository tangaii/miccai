# MedParse

**MedParse: A Task-Specialized Vision-Language Framework for Medical Image Parsing**

MedParse is a four-task medical image parsing pipeline built on a shared
MedGemma vision-language backbone. It maps one image and one task question to
one of four output contracts: a single option, a set of labels, image-space
boxes, or a bounded scalar. This repository contains the public inference graph,
component fitting utilities, input/output contracts, local diagnostic metrics,
and behavior tests. External model weights, adapters, fitted assets, retrieval
tables, challenge data, and learned heads are intentionally not distributed.

## Overview

The paper organizes the implementation into five modules:

1. **Shared MedGemma Representation Interface**
2. **Task-Routed Classification**
3. **Evidence-Guided Set Decoding**
4. **Frozen Spatial Query Decoding**
5. **Retrieval-Refined Quantile Regression**

The paper-to-code mapping is maintained in
[docs/METHOD_CODE_MAP.md](docs/METHOD_CODE_MAP.md). The external checkpoint
contract is maintained in [checkpoints/README.md](checkpoints/README.md), and
the public input schema is described in [data/README.md](data/README.md).

## Method

The runtime uses the external `google/medgemma-1.5-4b-it` model in separate
states. The raw base supplies projected image tokens and raw visual features;
the primary LoRA state supplies generation, multi-label scoring, and detection
features; the regression LoRA state supplies the generated numeric estimate.
Adapters are loaded one at a time and are never silently stacked.

- **Shared representation:** prompt rendering, greedy generation, decoder
  access, and projected image-token extraction.
- **Classification:** a route manifest selects a semantic image-token head,
  direct-prompt generation, or an instructional-generation fallback. Semantic
  inference is active for the released bone-marrow, fundus, and IUGC routes.
- **Multi-label classification:** generated proposals are refined with
  teacher-forced singleton evidence, candidate selection, listwise ranking,
  atom/cardinality probability models, a token-conditioned residual head, and
  cardinality-aware GFM decoding.
- **Detection:** primary-adapter image tokens and the final non-padding decoder
  state feed a two-layer spatial-query decoder. Coordinates are decoded
  directly rather than generated as text.
- **Regression:** adaptive image views, visual and intensity/Sobel geometry
  features, generated and visual estimates, cross-group retrieval, residual
  correction, and ordered quantile refinement are fused into a value clipped to
  `[0, 100]`.

## Environments and Requirements

### Installation

The supported installation path is:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
pip install -r requirements-dev.txt
```

`requirements.txt` mirrors the runtime dependencies declared in
`pyproject.toml`; `requirements-dev.txt` adds the test dependency.
This README follows the structure of the
[MICCAI Code Reproducibility Checklist](https://github.com/JunMa11/MICCAI-Reproducibility-Checklist#2-code-checklist-for-machine-learning-based-miccai-papers).

### Recorded environments

The release audit was run on September 21, 2026 in a Linux x86_64 development
environment with a Hygon C86-4G host, 256 logical CPUs, approximately 1.5 TiB
host memory, and a PPU-ZW810E device reported with 96 GiB memory. The recorded
software stack includes Python 3.12.3, NumPy 1.26.0, Pillow 12.2.0, PyYAML
6.0.3, SciPy 1.11.3, scikit-learn 1.3.2, joblib 1.1.1, PyTorch 2.9.0+ppu2.0.0,
Transformers 5.2.0, PEFT 0.18.0, Accelerate 1.12.0, safetensors 0.7.0,
CatBoost 1.2.10, pytest 7.2.0, and CUDA 12.9. These are development and
revalidation details, not a requirement that every user has the same
accelerator.

The paper reports a separate qualification run on an NVIDIA A10 over 1,783
inputs: 3,802 s (approximately 63.4 min), peak host-sampled GPU memory
21,797 MiB (approximately 21.3 GiB), and peak resident memory
19,554,471,936 bytes (approximately 18.2 GiB). The two environments should not
be conflated.

The external MedGemma model and LoRA adapters are required for real inference.
The public tests and dry-run smoke test do not download them.

## Dataset

Official FLARE challenge data and images are not redistributed in this
repository. Users must obtain the authorized data and follow the FLARE
organizer's access and licensing terms. No unverified download URL is provided
here.

The public inference contract is unlabeled JSONL (or a JSON list) with exactly
one local image reference per row:

```json
{"uid":"case-001","task_type":"classification","dataset":"fundus","prompt":"...","images":["/path/to/image.png"]}
```

Supported task types are `classification`, `multi_label_classification`,
`detection`, and `regression` (the parser also accepts documented aliases).
Every row requires a unique UID, a dataset/source name, a prompt or question,
and one image. Answer, target, label, reference, prediction, and related gold
fields are rejected by the inference validator. Training labels use a separate
contract and must not be passed to inference preparation.

The BUS-UCLM/BUSI ultrasound material is used by the released Detection study
as a separate 175-case labeled cohort. It is not described as the organizer's
hidden Detection split and is not combined with the 1,703-row validation-hidden
artifact.

## Preprocessing

The preprocessing behavior is defined in
`src/medical_parsing/schema.py` and the task modules:

- **Classification, multi-label classification, and Detection:** EXIF
  orientation is corrected, the image is converted to RGB, and the runtime
  prepares a deterministic `896 x 896` image using BICUBIC resampling before it
  is passed to the model processor.
- **Regression views:** wide or tall images use the original image plus three
  aspect-ratio crops with 15% overlap; near-square images use the original plus
  four quadrant crops. Each view is then processed by the same model path.
- **Regression geometry:** `geometry_one` converts the image to grayscale,
  resizes it to `896 x 896` with BILINEAR resampling, scales intensity by
  `255`, computes reflected-boundary Sobel derivatives, and pools intensity,
  horizontal-edge, and vertical-edge summaries into the fixed 960-dimensional
  descriptor.
- **Cropping:** not used as a separate step for classification, multi-label
  classification, or Detection; the regression adaptive-view policy is the
  documented exception.
- **Registration:** not used.
- **Additional intensity normalization:** not added by the repository beyond
  the explicit regression geometry scaling and the model processor's own tensor
  conversion.

Use `scripts/prepare_data.py` only to validate and normalize an unlabeled local
manifest. It is not a source-image preprocessing pipeline and it does not
produce learned assets:

```bash
python scripts/prepare_data.py \
  --input raw_manifest.jsonl \
  --output prepared.jsonl \
  --image-root /path/to/images
```

## Training and Fitting

`train.py` fits downstream components from user-owned labeled arrays or feature
caches. It does not create the route manifest, MLC template map, MLC candidate
library, or the final competition LoRA adapters. Those remain external or
data-derived assets. `src/medical_parsing/training/adapters.py` is a generic
LoRA helper and is not claimed to reproduce the final challenge adapters.

The public component producers are:

| Component | Required prepared inputs | Output |
| --- | --- | --- |
| `classification-head` | `tokens` NPZ array and semantic-label JSONL | `classification_heads.pt` |
| `detection-head` | `image_tokens`/`query_states` NPZ and normalized target JSON | `spatial_query_decoder.pt` |
| `multilabel-selector-ranker` | selector/ranker features, targets, and optional groups | selector and ranker CatBoost files |
| `multilabel-probability-models` | `features` and `[N,10,4]` targets | `multilabel_probability_models.joblib` |
| `multilabel-residual-head` | tokens, row features, base probabilities, targets | `multilabel_residual_head.pt` |
| `regression-visual-estimator` | visual features and scalar targets | `regression_visual_model.joblib` |
| `regression-reference` | visual features, geometry, targets, UIDs, groups | `regression_reference.joblib` |
| `regression-residuals` | UID-aligned residual values | `regression_residuals.npz` |
| `regression-quantile-head` | tokens, geometry, scalar targets | `regression_quantile_head.pt` |

Representative commands for every public component are shown below. Replace
the example filenames with files produced from authorized labeled data and
feature extraction:

```bash
python train.py --component classification-head \
  --features cls_tokens.npz --labels semantic_labels.jsonl \
  --output classification_heads.pt

python train.py --component detection-head \
  --features detection_features.npz --targets detection_targets.json \
  --output spatial_query_decoder.pt

python train.py --component multilabel-selector-ranker \
  --features mlc_candidates.npz \
  --output multilabel_candidate_selector.cbm \
  --secondary-output multilabel_candidate_ranker.cbm

python train.py --component multilabel-probability-models \
  --features mlc_probability.npz \
  --output multilabel_probability_models.joblib

python train.py --component multilabel-residual-head \
  --features mlc_residual.npz \
  --output multilabel_residual_head.pt

python train.py --component regression-visual-estimator \
  --features regression_visual.npz \
  --output regression_visual_model.joblib

python train.py --component regression-reference \
  --features regression_reference.npz \
  --output regression_reference.joblib

python train.py --component regression-residuals \
  --features regression_residuals.npz \
  --output fitted_regression_residuals.npz

python train.py --component regression-quantile-head \
  --features regression_quantile.npz \
  --output regression_quantile_head.pt
```

The detection feature and target producers are:

```bash
python tools/extract_detection_features.py \
  --input detection.jsonl \
  --base /path/to/medgemma \
  --adapter /path/to/primary-adapter \
  --output detection_features.npz

python tools/prepare_detection_targets.py \
  --input labeled_detection.jsonl \
  --output detection_targets.json
```

The frozen/default fitting contracts are explicit in the source and
`configs/default.yaml`. Key settings include three classification folds with
the semantic-head trainer defaults, CatBoost selector/ranker settings of 500
iterations, depth 7, and learning rate 0.05, 40 MLC probability models with
300 iterations and depth 6, 64-component regression PCA transforms, and the
Detection head's 30 epochs, batch size 256, AdamW learning rate `1e-3`, weight
decay `1e-4`, 5% warmup, cosine decay, and fixed spatial loss. Component-specific
contracts and required serialized keys are documented in
[checkpoints/README.md](checkpoints/README.md).

The `scripts/train_*.sh` files are thin compatibility wrappers around
`train.py`; they do not define a second training implementation.

## Inference

### Contract smoke test

This path validates all four task contracts without loading external models or
checkpoints:

```bash
python scripts/prepare_smoke_data.py --output-dir .smoke
python inference.py \
  --input .smoke/input.jsonl \
  --output .smoke/predictions.jsonl \
  --dry-run \
  --audit-json .smoke/audit.json
```

The smoke images are synthetic contract fixtures. They are not challenge data
and their outputs are not benchmark results.

### Real inference

Keep all learned assets outside the repository and pass them explicitly:

```bash
python inference.py \
  --input prepared.jsonl \
  --output predictions.jsonl \
  --checkpoint-dir /path/to/external/checkpoints \
  --base /path/to/medgemma \
  --adapter /path/to/primary-task-adapter \
  --reg-adapter /path/to/regression-task-adapter \
  --device cuda:0 \
  --audit-json run-audit.json
```

The base model and primary adapter are required for model-backed inference.
The regression adapter is required when regression rows are present. Detection
also requires `spatial_query_decoder.pt`; each task-specific branch requires
the assets listed in [checkpoints/README.md](checkpoints/README.md). The
runtime refuses missing assets and validates output UID order, task identity,
legal labels, finite boxes, and the `[0, 100]` regression range.

## Evaluation

`evaluate.py` is a local diagnostic evaluator for canonical predictions:

```bash
python evaluate.py \
  --reference labeled_reference.jsonl \
  --predictions predictions.jsonl \
  --output metrics.json
```

It reports classification accuracy; multi-label exact match, micro precision,
micro recall, micro-F1, and sample-F1; Detection micro precision/recall/F1 at
IoU 0.5; and regression MAE, RMSE, and bias. It does not download data, access
Codabench, or implement the organizer's official overall scorer. Official
challenge numbers must retain the organizer-provided evaluator and provenance.
The `scripts/inference.sh` and `scripts/evaluate.sh` files are likewise thin
convenience wrappers around the root CLIs.

## Results

The source-locked validation-hidden artifact contains 1,703 rows. The reported
validation evidence is:

| Evaluation source | Task | Rows | Metric | Result |
| --- | --- | ---: | --- | ---: |
| Source-locked validation-hidden artifact | Overall | 1,703 | overall score | 0.483753 |
| Source-locked validation-hidden artifact | Classification | 1,126 | balanced accuracy | 0.847607 |
| Source-locked validation-hidden artifact | Multi-label classification | 477 | micro-F1 | 0.526655 |
| Source-locked validation-hidden artifact | Regression | 100 | MAE | 11.987337 |
| Separate labeled ultrasound cohort | Detection | 175 | F1 at IoU 0.5 | 0.453552 |
| Separate labeled ultrasound cohort | Detection baseline | 175 | F1 at IoU 0.5 | 0.200542 |

The 175-case Detection result is a separate labeled cohort and is not part of
the 1,703-row denominator. Final testing-set evaluation remains pending and no
organizer leaderboard or final-test claim is made here.

## Reproducibility and External Assets

The repository is a code and contract release, not a standalone redistribution
of the challenge artifact. Exact fitted values require:

- authorized FLARE source data and source splits;
- the external `google/medgemma-1.5-4b-it` base model;
- the primary and regression PEFT adapters;
- route/template/candidate assets derived from the task data;
- fitted heads, estimators, retrieval tables, and residual files listed in
  [checkpoints/README.md](checkpoints/README.md).

The public release does not claim that `train.py` reconstructs the final
competition adapters or that `evaluate.py` reproduces an official organizer
score. It does provide the public component interfaces, serialization checks,
local diagnostics, and tests for the released prediction semantics.

## Repository Structure

```text
configs/                  Runtime defaults and reference component contracts
checkpoints/README.md     External asset names, producers, and schemas
data/README.md            Input schema and training-data boundary
docs/METHOD_CODE_MAP.md   Paper Method to code/symbol mapping
src/medical_parsing/      Package implementation
  data/                   Inference manifest preparation
  evaluation/             Local diagnostic metrics
  inference/              Four-task orchestration
  models/                 Backbone and neural modules
  tasks/                  Classification, MLC, Detection, and regression
  training/               Component fitting and generic LoRA helper
tests/                    Contract and behavior tests
inference.py              Public inference entry point
train.py                  Public fitting dispatcher
evaluate.py               Public local evaluator
```

## Contributing

Open an issue for a reproducibility problem or a narrowly scoped bug, and use
a pull request for code or documentation changes. Run `pytest -q` and the
smoke-test commands before opening a pull request. Do not commit challenge
data, model weights, adapters, fitted checkpoints, private keys, or local
machine paths.

## Citation

If you use this repository, cite:

```bibtex
@software{tang_medparse,
  author  = {Tang, Ai},
  title   = {MedParse: A Task-Specialized Vision-Language Framework for
             Medical Image Parsing},
  url     = {https://github.com/tangaii/MedParse},
  version = {0.1.0}
}
```

No DOI or final proceedings metadata is asserted here. The machine-readable
metadata is in [CITATION.cff](CITATION.cff).

## Acknowledgement

We acknowledge the FLARE 2026 organizers and data contributors, the MedGemma
upstream model and documentation, and the PyTorch, Transformers, PEFT, SciPy,
scikit-learn, CatBoost, and related open-source communities.

## License

The original research code and refactoring work are released under the MIT
License; see [LICENSE](LICENSE) and [LICENSE_DECISION.md](LICENSE_DECISION.md).
Upstream model, adapter, dependency, and dataset terms remain applicable to
external artifacts.
