# MedParse

**MedParse: A Task-Specialized Vision-Language Framework for Medical Image Parsing**

MedParse is a four-task medical image parsing pipeline built on a shared
MedGemma vision-language backbone. One image and one task question are mapped
to a classification option, a label set, image-space boxes, or a bounded scalar.
This repository is the public implementation accompanying the paper.

Useful references:

- [GitHub repository](https://github.com/tangaii/MedParse)
- [Checkpoint contract](checkpoints/README.md)

## Environments and Requirements

| Component | Setting |
| --- | --- |
| OS | Linux x86_64 |
| Python | 3.12.3 |
| Development accelerator | PPU-ZW810E, 96 GiB |
| Qualification accelerator | NVIDIA A10 |
| CUDA | 12.9 |
| Dependencies | `requirements.txt`, `pyproject.toml` |

Install the runtime and development dependencies with:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
pip install -e ".[dev]"
```

The qualification run used an NVIDIA A10 for 1,783 inputs and completed in
3,802 s (about 63.4 min). Peak host-sampled GPU memory was 21,797 MiB (about
21.3 GiB); peak resident memory was 19,554,471,936 bytes (about 18.2 GiB).

## Dataset

FLARE challenge data and images are obtained by the user under the organizer's
access and licensing terms; they are not included here. The public inference
input is unlabeled JSONL (a JSON list is also accepted), with one local image
per row:

```json
{"uid":"case-001","task_type":"classification","dataset":"fundus","prompt":"...","images":["/path/to/image.png"]}
```

Supported task types are `classification`, `multi_label_classification`,
`detection`, and `regression`. Each row has a unique UID, a dataset/source,
one question or prompt, and exactly one image. Training labels use a separate
contract. Inputs must not include answer, target, label, reference, or
prediction fields; detection predictions use original-image `[x1,y1,x2,y2]`
boxes. `scripts/prepare_data.py` performs manifest validation and normalization.

The BUS-UCLM/BUSI ultrasound data form a separate 175-case labeled cohort for
the Detection study. This cohort is not the organizer's hidden Detection split
and is not included in the 1,703-row validation-hidden artifact.

## Preprocessing

- Classification, multi-label classification, and Detection correct EXIF
  orientation, convert to RGB, and resize to `896 x 896` with BICUBIC
  resampling.
- Regression uses the original image plus three 15%-overlap aspect-ratio crops
  for wide or tall images, and four quadrant crops for near-square images.
- The regression geometry descriptor converts to grayscale, uses `896 x 896`
  BILINEAR resampling, scales intensity by `255`, computes reflected-boundary
  Sobel derivatives, and produces a fixed 960-dimensional summary.
- Cropping is task-specific as above; registration is not used.

Prepare an unlabeled local manifest with:

```bash
python scripts/prepare_data.py \
  --input raw_manifest.jsonl \
  --output prepared.jsonl \
  --image-root /path/to/images
```

Manifest preparation validates and normalizes references; it is not source
image preprocessing and does not create learned assets.

## Method Overview

The paper-level implementation has five modules:

1. **Shared MedGemma Representation Interface** provides prompt rendering,
   greedy generation, decoder access, and projected image-token features.
2. **Task-Routed Classification** selects a semantic image-token head or a
   generation route from a task manifest.
3. **Evidence-Guided Set Decoding** refines generated proposals with singleton
   evidence, candidate ranking, probability models, and cardinality-aware F1
   decoding.
4. **Frozen Spatial Query Decoding** predicts Detection boxes from image tokens
   and decoder states rather than generating coordinates as text.
5. **Retrieval-Refined Quantile Regression** fuses visual, geometry, generated,
   retrieval, residual, and ordered-quantile estimates into a value in
   `[0, 100]`.

| Paper module | Primary code |
| --- | --- |
| Shared interface | `src/medical_parsing/models/backbone.py` |
| Classification | `src/medical_parsing/tasks/classification.py` |
| Evidence-guided set decoding | `src/medical_parsing/tasks/multilabel.py` |
| Spatial query decoding | `src/medical_parsing/tasks/detection.py` |
| Quantile regression | `src/medical_parsing/tasks/regression.py` |

## Training

`train.py` fits downstream components from user-owned labeled arrays or feature
caches. The public components are:

| Component | Input | Output |
| --- | --- | --- |
| `classification-head` | token NPZ + semantic-label JSONL | `classification_heads.pt` |
| `detection-head` | image-token/query-state NPZ + normalized targets | `spatial_query_decoder.pt` |
| `multilabel-selector-ranker` | selector/ranker features and targets | selector and ranker CatBoost files |
| `multilabel-probability-models` | features + `[N,10,4]` targets | `multilabel_probability_models.joblib` |
| `multilabel-residual-head` | tokens, row features, probabilities, targets | `multilabel_residual_head.pt` |
| `regression-visual-estimator` | visual features + scalar targets | `regression_visual_model.joblib` |
| `regression-reference` | visual/geometry features + targets, UIDs, groups | `regression_reference.joblib` |
| `regression-residuals` | UID-aligned residual values | `regression_residuals.npz` |
| `regression-quantile-head` | tokens, geometry, scalar targets | `regression_quantile_head.pt` |

Use the component-specific pattern below; options are shown for representative
classification, Detection, multi-label, and regression fits:

```bash
python train.py --component <component> [component options]

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

python train.py --component regression-reference \
  --features regression_reference.npz \
  --output regression_reference.joblib
```

Detection fitting defaults are 30 epochs, batch size 256, AdamW with learning
rate `1e-3` and weight decay `1e-4`, 5% warmup, and cosine decay. See
`python train.py --help` and [checkpoints/README.md](checkpoints/README.md) for
all component options, serialized keys, and feature producers. The generic
`src/medical_parsing/training/adapters.py` helper does not claim to reproduce
the final competition LoRA adapters.

## Inference

Run the contract smoke test without external models or checkpoints:

```bash
python scripts/prepare_smoke_data.py --output-dir .smoke
python inference.py \
  --input .smoke/input.jsonl \
  --output .smoke/predictions.jsonl \
  --dry-run \
  --audit-json .smoke/audit.json
```

For model-backed inference, pass the external base model, adapters, and
checkpoint directory explicitly:

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

The required filenames and task-specific assets are listed in
[checkpoints/README.md](checkpoints/README.md). The runtime validates UID order,
task identity, legal labels, finite boxes, and the `[0, 100]` regression range.

## Evaluation

Evaluate canonical predictions locally with:

```bash
python evaluate.py \
  --reference labeled_reference.jsonl \
  --predictions predictions.jsonl \
  --output metrics.json
```

The evaluator reports classification accuracy; multi-label exact match,
micro-F1, and sample-F1; Detection F1 at IoU 0.5; and regression MAE, RMSE,
and bias. It is a local diagnostic evaluator, not the organizer's official
scorer.

## Results

The source-locked validation-hidden artifact contains 1,703 rows.

| Source | Task | Rows | Metric | Result |
| --- | --- | ---: | --- | ---: |
| Validation-hidden artifact | Overall | 1,703 | overall score | 0.483753 |
| Validation-hidden artifact | Classification | 1,126 | balanced accuracy | 0.847607 |
| Validation-hidden artifact | Multi-label classification | 477 | micro-F1 | 0.526655 |
| Validation-hidden artifact | Regression | 100 | MAE | 11.987337 |
| Separate labeled ultrasound cohort | Detection | 175 | F1 at IoU 0.5 | 0.453552 |
| Separate labeled ultrasound cohort | Detection baseline | 175 | F1 at IoU 0.5 | 0.200542 |

The 175-case Detection result is a separate cohort and is not part of the
1,703-row denominator. Final testing-set evaluation remains pending.

## Reproducibility Notes

Exact fitted values additionally require authorized FLARE data and source
splits, the `google/medgemma-1.5-4b-it` base model, primary and regression
PEFT adapters, data-derived route/template/candidate assets, and the fitted
heads, estimators, retrieval tables, and residual files listed in
[checkpoints/README.md](checkpoints/README.md). These external assets and
challenge images are not committed to the repository. Official challenge scores
require the organizer-provided evaluator.

The checkpoint directory expects the base model and adapters externally, plus
the route manifest, classification heads, MLC templates/library/selector/
ranker/probability/residual assets, `spatial_query_decoder.pt`, and regression
visual/reference/residual/quantile assets. The checkpoint README is the filename
and schema contract for this bundle.

## Repository Structure

```text
configs/default.yaml      Runtime defaults and reference contracts
checkpoints/README.md     External asset names and schemas
src/medical_parsing/      Package implementation
  data/                   Manifest preparation
  evaluation/             Local metrics
  inference/              Four-task orchestration
  models/                 Backbone and neural modules
  tasks/                  Task implementations
  training/               Component fitting
tests/                    Contract and behavior tests
scripts/                  Manifest and smoke-data utilities
tools/                    Feature and target preparation utilities
inference.py              Public inference CLI
train.py                  Public fitting CLI
evaluate.py               Local evaluation CLI
```

## Contributing

Open an issue for a reproducibility problem or a narrowly scoped bug. Before a
pull request, run `pytest -q` and the smoke-test commands. Do not commit
challenge data, model weights, adapters, fitted checkpoints, private keys, or
local machine paths.

## Citation

```bibtex
@software{tang_medparse,
  author  = {Tang, Ai},
  title   = {MedParse: A Task-Specialized Vision-Language Framework for
             Medical Image Parsing},
  url     = {https://github.com/tangaii/MedParse},
  version = {0.1.0}
}
```

Machine-readable citation metadata is provided in [CITATION.cff](CITATION.cff).

## Acknowledgement

We thank the FLARE 2026 organizers and data contributors and acknowledge the
MedGemma model and the open-source software used in this work.

## License

The original research code is released under the MIT License; see
[LICENSE](LICENSE). External model, adapter, dependency, and dataset terms
remain applicable.
