# R3-MedGemma

Routing, Refinement, and Retrieval for Heterogeneous Medical Image Parsing

Public research code for a three-task medical-image parsing pipeline. The
repository contains the readable inference graph, component-wise fitting
helpers, input/output contracts, local diagnostic metrics, and behavior tests.
The MedGemma base model, LoRA adapters, fitted estimators, retrieval tables,
challenge data, and learned heads are external inputs and are not
distributed here.

## Paper overview

R3-MedGemma is the paper name for the frozen four-module implementation:
shared MedGemma backbone, task-routed semantic--generative classification,
structured evidence-refined multi-label prediction, and multi-view
retrieval--quantile regression. The locked development/frozen-validation
artifact reports overall 0.483, classification balanced accuracy 0.846,
multi-label F1 0.527, and regression MAE 11.99. These are frozen validation
evidence; testing-set evaluation is not reported.

The paper title is ``R3-MedGemma: Routing, Refinement, and Retrieval for
Heterogeneous Medical Image Parsing''. The public repository contains the
implementation and paper-to-code map; external weights, fitted heads,
retrieval tables, and source data remain outside the repository.

## Method overview

The method uses one external google/medgemma-1.5-4b-it vision-language
backbone. The final input contract is exactly one local image per row. Images
receive EXIF correction, RGB conversion, and deterministic 896 x 896 BICUBIC
preparation. Task consumers are loaded sequentially; raw base-model use, the
primary LoRA adapter, and the regression LoRA adapter are separate model
states and are never silently stacked.

The paper-level modules are:

1. Unified Framework and Shared MedGemma Backbone — prompt rendering,
   generation, raw projected image-token extraction, decoder access, and
   single-adapter loading.
2. Task-Routed Semantic–Generative Classification — a route manifest selects
   the semantic image-token head, direct prompt generation, or instructional
   generation fallback.
3. Structured Evidence-Refined Multi-Label Prediction — an initial generated
   set is refined using parser-native singleton evidence, candidate selection,
   listwise re-ranking, atom/cardinality probability models, a
   token-conditioned residual head, and GFM decoding.
4. Multi-View Retrieval–Quantile Regression — adaptive image views, visual and
   intensity-edge geometry representations, visual and generated estimates,
   cross-group retrieval, residual correction, and a spatial quantile head are
   fused into the bounded numeric output.

The complete paper-to-code table is in
[docs/METHOD_CODE_MAP.md](docs/METHOD_CODE_MAP.md). The README and that table
use the same formal names; compatibility aliases are called out where they
preserve the original Python API or serialized feature contract.

## Unified Framework and Shared MedGemma Backbone

load_raw_bundle loads the external base model for raw visual features.
load_adapter_bundle loads exactly one PEFT LoRA adapter for one task state,
checks the declared base identity, detects source and target visual-tower
namespaces independently, remaps keys only when their orientations are
actually opposite, and reports total/loaded/missing/unexpected adapter keys.

The runtime therefore follows this state contract:

- raw base state: used for semantic image-token extraction and regression
  multi-view visual features;
- primary adapter state: used for classification generation, multi-label
  generation/scoring, and the regression spatial-token branch;
- regression adapter state: used only for the generated numeric regression
  estimate.

extract_image_tokens calls model.get_image_features(pixel_values=...). Its
compatibility prompt argument is retained for the unchanged processor batch,
but it does not condition the returned representation. These are projected
image tokens, not two prompt-conditioned semantic/spatial feature families.

Generation is greedy (do_sample=False) with the configured token limit.
set_determinism enables the available random, cuDNN, and PyTorch determinism
controls. PyTorch is called with warn_only=True, so this is a
determinism-control setting rather than a claim of bitwise identity on every
hardware/software stack.

## Task-Routed Semantic–Generative Classification

The route manifest is normalized to these formal internal routes:

- semantic_head — SemanticImageTokenHead over raw [256, 2560] image tokens,
  with three fitted folds averaged before option mapping;
- direct_prompt_generation — prompt_only_generated_text, which sends the
  system prompt and the row's original prompt directly;
- instructional_generation_fallback — generated_text, which adds the task
  instruction and any structured options before generation.

The manifest parser also accepts the historical values semantic, prompt, and
fallback; the external manifest does not need to be rewritten. Dental semantic
heads remain in SemanticImageTokenHead for strict checkpoint compatibility, but
the final classification orchestration activates semantic inference only for
bone_marrow, fundus, and iugc. A dental row is not silently treated as an
active semantic-head route; it uses the generation fallback when the manifest
requests the legacy semantic route.

resolve_classification_slot, map_option_to_semantic_concept, and
map_semantic_concept_to_option keep the semantic ontology separate from
row-specific option letters. Every output is checked against the legal options
parsed from that row.

## Structured Evidence-Refined Multi-Label Prediction

The fixed parser vocabulary contains ten legal atoms. The formal atom groups
are SEMANTIC_LABEL_ATOMS and AUXILIARY_LABEL_ATOMS; the old SEMANTIC and PSEUDO
names remain compatibility aliases only. The canonical set parser is
schema.parse_label_set; parse_multilabel is its compatibility alias.
Serialization uses the fixed atom order and a trailing semicolon.

The inference chain is:

1. Initial Generative Label Proposal — generated_text produces the initial
   answer set.
2. Teacher-Forced Singleton Evidence Scoring —
   score_teacher_forced_streaming scores the ten serialized singleton answers
   using the actual chat template and image processor.
3. Thresholded Candidate Refinement — build_initial_candidate_table,
   build_candidate_selector_features, and select_refined_candidate implement
   deterministic keep/add/drop and nearby replacement candidates with the
   frozen 0.05 change threshold.
4. Listwise Candidate Re-ranking — build_reranked_candidates and
   build_candidate_ranker_features construct and rank the corrected set.
   apply_tooth_position_aware_correction restores the frozen KEEP/PNSS choice
   for FDI tooth position 8 when the ranker selects a non-KEEP candidate.
5. Label–Cardinality Joint Probability Estimation —
   build_probability_model_features feeds the ten-atom by four-cardinality
   fitted probability models.
6. Token-Conditioned Residual Probability Refinement —
   MultiLabelResidualProbabilityHead adds the bounded learned residual to the
   base probabilities.
7. Cardinality-Aware GFM Decoding — gfm_decode evaluates prediction
   cardinalities 1 through 10 using the fixed F1 utility and deterministic
   ties.

The normal scorer materializes decoder hidden states once and visits the
language-model head in vocab_chunk_size=16 vocabulary rows. The row batch size
is an MLC setting (scoring_row_batch_size=4). The full-logit path is only a
capability fallback for wrappers that do not expose the decoder and uses the
same final-logit soft-cap when one is declared. Alignment, shape, non-finite,
and model errors are not silently converted into that fallback.

The public implementation and a final deployment may use different but
algebraically equivalent vocabulary-chunk schedules. The candidate-score
definition and prediction semantics remain unchanged.

## Multi-View Retrieval–Quantile Regression

build_adaptive_image_views keeps the frozen crop policy: wide and tall images
produce the original plus three crops; near-square images produce the original
plus four quadrant crops. The implementation is aspect-ratio adaptive; the
crop algorithm is not changed by this documentation.

Each view yields projected 2,560-dimensional visual features. The original,
mean-crop, and max-crop views form a 7,680-dimensional visual representation.
The 960-dimensional geometry descriptor combines normalized intensity and
Sobel edges from 16 x 16 pooled cells, 32 vertical bands, and 32 horizontal
bands for each of three channels.

The formal numerical chain is:

~~~
generated_numeric_estimate = numeric generation branch
visual_regression_estimate = fitted visual estimator
base_fused_estimate = 0.5 * generated_numeric_estimate
                       + 0.5 * visual_regression_estimate
retrieval_estimate = cross-group weighted-median retrieval
retrieval_refined_estimate = 0.75 * base_fused_estimate
                             + 0.25 * retrieval_estimate
retrieved_residual_correction = cross-group residual retrieval
residual_corrected = clip(retrieval_refined_estimate
                          + 0.5 * retrieved_residual_correction, 0, 100)
prediction = clip(0.75 * residual_corrected
                  + 0.25 * upper_quantile_estimate, 0, 100)
~~~

SpatialQuantileRefinementHead uses 256 coordinate-aware tokens, a 128-D
projection, 2-D coordinates, four learned queries, and ordered Q25/Q50/Q75
outputs. The fixed/default regression constants are centralized in
RegressionConfig.

## Input and output contract

Input records are unlabeled JSONL/JSON objects. Each row requires a unique
UID, task type, dataset/source, prompt/question, and exactly one local image
reference:

~~~
{"uid":"case-001","task_type":"classification","dataset":"fundus","prompt":"...","images":["/local/image.png"]}
~~~

Local paths and validated zip://archive::member references are supported;
remote URLs are rejected. Answer, target, label, reference, prediction, and
related gold fields are rejected by the inference validator. Training data
has a separate labeled contract and must not be passed through the inference
preparation script.

Prediction output is canonical JSONL with exactly:

~~~
{"uid":"case-001","task_type":"classification","prediction":"A"}
~~~

The output validator checks UID order, task identity, legal classification
letters, legal multi-label atoms, finite regression values, and the [0, 100]
regression range.

## Quick start

Create an isolated environment and install runtime dependencies before the
editable package:

~~~
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pip install -r requirements-dev.txt  # tests only
~~~

Run the public contract smoke test without external model or checkpoint
assets:

~~~
python scripts/prepare_smoke_data.py --output-dir .smoke
python inference.py --input .smoke/input.jsonl \
  --output .smoke/predictions.jsonl --dry-run
pytest -q
~~~

The code was revalidated on September 1, 2026 in the project environment with
Python 3.12.3, NumPy 1.26.0, Pillow 12.2.0, PyYAML 6.0.3, SciPy 1.11.3,
scikit-learn 1.3.2, joblib 1.1.1, PyTorch 2.9.0+ppu2.0.0, Transformers 5.2.0,
PEFT 0.18.0, Accelerate 1.12.0, safetensors 0.7.0, CatBoost 1.2.10, and
pytest 7.2.0. CUDA was available with CUDA 12.9. Other stacks may produce
small floating-point differences.

For actual inference, keep all learned assets outside this repository and
pass their directory explicitly:

~~~
python inference.py \
  --input prepared.jsonl \
  --output predictions.jsonl \
  --checkpoint-dir /path/to/external/checkpoints \
  --base /path/to/medgemma \
  --adapter /path/to/primary-task-adapter \
  --reg-adapter /path/to/regression-task-adapter \
  --device cuda:0 \
  --audit-json run-audit.json
~~~

The regression adapter is required only when regression rows are present. See
checkpoints/README.md for the exact asset contract.

## Training and fitting

The public fitting APIs cover the following components:

- classification-head — three-fold semantic image-token heads;
- multilabel-selector-ranker — candidate selector and listwise ranker;
- multilabel-probability-models — 40 atom/cardinality probability models;
- multilabel-residual-head — token-conditioned residual probability head;
- regression-visual-estimator — visual estimator with scaler/PCA;
- regression-reference — cross-group retrieval table and transforms;
- regression-residuals — UID-aligned cross-fitted residual table;
- regression-quantile-head — spatial quantile head and geometry transforms.

Use the explicit component CLI so the command names the paper module being fit:

~~~
python train.py --component classification-head \
  --features tokens.npz --labels semantic_labels.jsonl \
  --output classification_heads.pt

python train.py --component regression-visual-estimator \
  --features visual_features.npz --output regression_visual_model.joblib
~~~

The legacy --task interface remains supported and infers a component from NPZ
keys. The component dispatcher does not create the route manifest, template
map, candidate library, or LoRA adapters: those are external or data-derived
preparation steps documented in the checkpoint contract.

training/adapters.py is a generic adapter-training utility. Its target
modules, prompt supervision, and training arguments are configurable helper
defaults; the repository does not claim that they exactly reproduce the
external challenge adapters.

The files under configs/classification.yaml, configs/multilabel.yaml, and
configs/regression.yaml are reference-only component contracts. Runtime
defaults are read from configs/default.yaml; load_config is the single runtime
configuration source.

## Evaluation

evaluate.py reports diagnostic/local classification accuracy, multi-label exact
match and micro/sample F1, and regression MAE/RMSE/bias:

~~~
python evaluate.py --reference labeled_reference.jsonl \
  --predictions predictions.jsonl --output metrics.json
~~~

This is not the organizer's official challenge evaluator: it does not report
balanced accuracy or the challenge overall score, does not download data, and
does not contact Codabench. Official paper/challenge numbers must be computed
with the organizer-provided evaluator and its documented provenance.

## Scope and limitations

The repository is a readable implementation and reproducibility contract, not a
redistribution of the learned challenge artifact. Exact fitted values require the
corresponding labeled source data, source splits, external MedGemma
model/adapters, fitted estimators, and retrieval tables. The public code does
does not claim to reproduce an organizer score from this repository alone.

Only source code, contracts, and tests are included; learned weights and
restricted source data remain external.

## Repository layout

~~~
configs/                  Runtime defaults and reference component contracts
checkpoints/README.md     External asset contract; no weights are included
data/README.md            Inference data layout and training-data boundary
docs/METHOD_CODE_MAP.md   Paper Method to code/symbol mapping
src/medical_parsing/      Package code
  data/                   Inference manifest preparation
  evaluation/              Diagnostic/local metrics
  inference/              Three-branch orchestration
  models/                 Backbone and paper-locatable neural modules
  tasks/                  Classification, multi-label, and regression logic
  training/               Component fitting and generic LoRA utility
tests/                    Contract and behavior tests
inference.py              Public inference entry point
train.py                  Public component fitting dispatcher
evaluate.py               Public diagnostic evaluation entry point
~~~

## License

The refactored source is released under the MIT License. Upstream model,
adapter, dependency, and dataset terms remain applicable to external
artifacts; see LICENSE_DECISION.md.
