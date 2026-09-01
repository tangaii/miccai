# Medical Image Parsing

Initial public research release for a three-task medical-image parsing
pipeline. The repository contains the readable inference, training/fitting,
data-preparation, evaluation, and contract-test code. The trained model
weights, task adapters, fitted estimators, competition data, and private
retrieval tables are external inputs and are intentionally not distributed.

## Method overview

All three branches share one external MedGemma 1.5 4B instruction-tuned
backbone. The image path uses EXIF correction, RGB conversion, and deterministic
896 x 896 BICUBIC preparation. The final method uses a small number of
task-specific consumers of the same backbone:

- classification combines a semantic image-token head with a prompt-generation
  fallback selected by an external route manifest;
- multi-label classification uses a generated initial set, parser-native
  teacher-forced singleton scores, constrained candidate correction, candidate
  ranking, probability priors, a learned residual head, and a cardinality-aware
  decoder;
- regression combines multi-view visual features, fixed geometry features,
  cross-group retrieval, a visual estimator, a numeric generation estimate, a
  residual correction, and a spatial quantile head.

The public code keeps these contracts explicit. File paths and command-line
interfaces are refactored for portability, while prompt rendering,
tokenization, image processing, candidate ordering, feature dimensions,
normalization, decoder ties, and output serialization remain part of the
method contract.

## Shared backbone

The expected base model is `google/medgemma-1.5-4b-it`, supplied through a
local Hugging Face directory. The adapter loader accepts one LoRA adapter at a
time, verifies its base-model identity, and rejects unexpected adapter stacks.
The classification, multi-label, and regression adapters are loaded
sequentially so the base model is never silently combined with incompatible
task adapters.

The processor is configured for right padding. Generation is greedy and
deterministic. Image-token consumers expect `[batch, 256, 2560]` projected
vision tokens.

## Classification pipeline

The route manifest maps normalized dataset/question keys to one of three public
route names: `semantic`, `prompt`, or `fallback`.

For eligible bone-marrow, fundus, IUGC, and dental rows, semantic inference
uses the raw backbone image tokens. A LayerNorm and 2560-to-128 projection
feed quantile, mean, and standard-deviation summaries. Source-specific trunks
produce the diagnosis, age, plane, or dental jaw outputs. Dental upper and
lower halves share the jaw heads. Three fitted folds are averaged before the
semantic concept is mapped back to the row's option letter.

Prompt and fallback routes use the external LoRA adapter and the same
parser-safe option mapping. Every input row must receive exactly one legal
option letter.

## Multi-label pipeline

The fixed vocabulary contains ten parser atoms. Generated answers are parsed
into sets and serialized in a deterministic atom order with a trailing
semicolon. For each row, the scorer evaluates the ten singleton serialized
answers using the actual chat template and image processor.

The post-processing sequence is:

1. generate the initial set;
2. build a deterministic keep/add/drop/nearby-replacement candidate table;
3. score candidates with a fitted selector and apply the configured 0.05
   change threshold;
4. rank the corrected candidates with a fitted ranker;
5. fit forty atom/cardinality prior models and combine them with the image
   residual head;
6. decode cardinalities 1 through 10 using the fixed
   `2 / (predicted_cardinality + gold_cardinality)` utility.

The candidate library, selector, ranker, probability models, and residual head
are all external checkpoint assets. Their descriptive filenames and expected
consumers are documented in `checkpoints/README.md`.

## Regression pipeline

Each image yields the original view plus three aspect-ratio-aware crops. Each
view is processed by the raw backbone vision tower and multimodal projector;
the mean pooled 2560-dimensional view features are combined into a 7680-
dimensional original/mean-crop/max-crop representation.

The geometry descriptor has 960 values from normalized intensity and Sobel
edges: 16 x 16 pooled cells, 32 vertical bands, and 32 horizontal bands for
each of the three channels. Visual and geometry features use persisted
StandardScaler/PCA transforms, L2 normalization, and a 0.5/0.5 concatenation
for cross-group cosine retrieval.

The final estimate is the clipped blend

```text
base = 0.5 * generated_numeric_value + 0.5 * visual_estimator
retrieval_blend = 0.75 * base + 0.25 * cross_group_weighted_median
residual_corrected = clip(retrieval_blend + 0.5 * residual_median, 0, 100)
prediction = clip(0.75 * residual_corrected + 0.25 * spatial_q75, 0, 100)
```

The spatial head uses four learned queries over coordinate-aware 16 x 16
tokens and returns ordered 0.25, 0.50, and 0.75 estimates.

## Memory-efficient scoring

The normal multi-label scorer runs the decoder once and obtains hidden states
without constructing a full vocabulary-logit tensor. It visits the language
model head in `TOKEN_CHUNK=16` vocabulary rows, accumulates the log-sum-exp
denominator, and evaluates only the target token rows. The operational row
batch is four rows by ten singleton answers. This keeps the peak vocabulary
activation proportional to `target_tokens x 16` rather than
`batch x sequence_length x vocabulary_size`.

`score_logits_full` is retained only as a small reference/compatibility path;
the tests compare it with `score_hidden_streaming` for numerical equivalence.

## Repository layout

```text
configs/                  Runtime and task configuration
checkpoints/README.md      External asset contract; no weights are included
data/README.md             Input layout and redistribution policy
src/medical_parsing/      Package code
  data/                   Input normalization
  evaluation/             Classification, set, and regression metrics
  inference/              Three-branch orchestration
  models/                 Backbone and fitted-head definitions
  tasks/                  Classification, multi-label, and regression logic
  training/               Head, estimator, residual, and LoRA fitting helpers
scripts/                  Data, smoke, training, inference, and evaluation CLIs
tools/                    Feature extraction helper
tests/                    Contract and behavior tests
inference.py              Public inference entry point
train.py                  Public fitting dispatcher
evaluate.py               Public evaluation entry point
```

## Input and output

Input records are unlabeled JSONL objects with at least:

```json
{"uid":"case-001","task_type":"classification","dataset":"fundus","prompt":"...","images":["/local/image.png"]}
```

`images` can contain local paths or validated `zip://archive::member`
references. Remote URLs are rejected. Every UID must be unique. Prediction
output is canonical JSONL with exactly these keys:

```json
{"uid":"case-001","task_type":"classification","prediction":"A"}
```

The output validator checks UID order, task identity, legal classification
letters, legal multi-label atoms, finite regression values, and the `[0, 100]`
regression range.

## Quick start

Create an isolated environment and install the package:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Create a synthetic fixture and validate the full input contract without
external assets:

```bash
python scripts/prepare_smoke_data.py --output-dir .smoke
python inference.py --input .smoke/input.jsonl --output .smoke/predictions.jsonl --dry-run
pytest -q
```

For actual inference, place the canonical fitted files named in
`checkpoints/README.md` in an external directory and provide the base model
and two task adapters:

```bash
python inference.py \
  --input prepared.jsonl \
  --output predictions.jsonl \
  --checkpoint-dir /path/to/external/checkpoints \
  --base /path/to/medgemma \
  --adapter /path/to/official-task-adapter \
  --reg-adapter /path/to/regression-task-adapter \
  --device cuda:0 \
  --audit-json run-audit.json
```

The regression adapter is required only when regression rows are present. The
same command supports any mixture of the three task types.

## Training and fitting

The repository includes code for:

- extracting shared image tokens with `tools/extract_features.py`;
- fitting three-fold semantic classification heads;
- fitting candidate selector/ranker models and forty probability models;
- training the multi-label residual head;
- fitting the visual regression estimator and retrieval reference;
- writing UID-aligned residuals;
- training the spatial regression quantile head;
- training one LoRA adapter with the external base model.

The root dispatcher selects a component based on the NPZ keys. Examples:

```bash
python train.py --task classification --features tokens.npz \
  --labels semantic_labels.jsonl --output classification_heads.pt
python train.py --task regression --features visual_features.npz \
  --output regression_visual_model.joblib
```

Fitting requires the original labeled training tables or user-owned
replacement data. Those data, feature caches, fitted estimators, and model
weights are not part of this repository.

## Evaluation

Evaluate a canonical prediction file against a labeled JSONL supplied by the
user:

```bash
python evaluate.py --reference labeled_reference.jsonl \
  --predictions predictions.jsonl --output metrics.json
```

The evaluator reports classification accuracy, multi-label exact match and
micro/sample F1, and regression MAE/RMSE/bias. It does not download data or
contact a competition service.

## Scope and limitations

This is an initial research release of the method implementation. It does not
claim a paper publication, a leaderboard position, or a final competition
score. The exact fitted asset packs are external because they contain learned
parameters, fitted transforms, retrieval references, or data-derived tables.
The fitting APIs cover their public serialization contracts, but reproducing
the original fitted values requires the corresponding labeled training data,
source splits, and external base/adapters.

The repository contains no Docker image, submission archive, validation
images/labels, prediction cache, private dataset, or trained weight file.

## License

The refactored source is released under the MIT License. Upstream model,
adapter, dependency, and dataset terms remain applicable to those external
artifacts; see `LICENSE_DECISION.md`.
