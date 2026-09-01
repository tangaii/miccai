# Method–Code Map

This document is the canonical paper-to-code map for the frozen public
implementation.  It describes runtime behavior, not experiment history.
Implementation details are separated from the method modules so that
reproducibility plumbing is not presented as an algorithmic contribution.

## Module map

| Paper Method Section | Formal Module Name | Purpose | Inference File | Training File | Key Class | Key Function | Input | Output | External Assets | Config | Core Method / Implementation Detail |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3.1 | Shared MedGemma Vision-Language Backbone | Shared multimodal representation, chat rendering, generation, and decoder access | src/medical_parsing/models/backbone.py | src/medical_parsing/training/adapters.py | external MedGemma model | load_raw_bundle, load_adapter_bundle, extract_image_tokens, decoder_hidden_states, generated_text | one validated local image plus task prompt | projected image tokens, generated text, or decoder hidden states | external base model and one adapter per loaded task state | ModelConfig; RegressionConfig dimensions | Model/generation calls are core; offline flags, CUDA checks, cleanup, and audits are engineering |
| 3.2 | Task-Routed Semantic–Generative Classification | Route each row to semantic image-token classification, direct prompt generation, or instructional fallback | src/medical_parsing/tasks/classification.py | src/medical_parsing/training/classification.py | SemanticImageTokenHead | run_classification, resolve_classification_slot, map_semantic_concept_to_option | raw image tokens or generated text plus row options | one legal option letter | classification route manifest; three-fold classification_heads.pt; primary adapter for generation | three folds; 2560-to-128 head contract | Route/ontology/option mapping is method behavior; route counts and model cleanup are implementation details |
| 3.3 | Initial Generative Label Proposal | Produce the initial multi-label answer set | src/medical_parsing/tasks/multilabel.py | external adapter preparation | external MedGemma + primary adapter | generated_text | one image and MLC prompt | raw generated label text | primary adapter | ModelConfig generation defaults | generation is method behavior; batch/device handling is engineering |
| 3.4 | Teacher-Forced Singleton Evidence Scoring | Score the ten legal serialized singleton answers under the actual chat template | src/medical_parsing/tasks/multilabel.py | feature preparation is external | external decoder/language-model head | score_teacher_forced_streaming, score_hidden_streaming | decoder hidden states, vocabulary head, singleton token mapping | row-by-ten score matrix | primary adapter | vocab_chunk_size=16; MLC scoring_row_batch_size=4 | streaming/full-logit math is method-supporting inference behavior; capability fallback is engineering |
| 3.5 | Thresholded Candidate Refinement | Construct parser-native keep/add/drop/nearby-replacement candidates and select a changed set only above threshold | src/medical_parsing/tasks/multilabel.py | src/medical_parsing/training/multilabel.py | fitted CatBoost regressor | build_initial_candidate_table, build_candidate_selector_features, select_refined_candidate | initial set, singleton scores, metadata, candidate library | refined candidate set | candidate library; candidate selector | candidate_change_threshold=0.05; max_replacement_candidates=32 | candidate order/tie behavior is frozen method behavior; array materialization is implementation |
| 3.6 | Listwise Candidate Re-ranking | Build the second candidate list and rank the refined alternatives | src/medical_parsing/tasks/multilabel.py | src/medical_parsing/training/multilabel.py | fitted CatBoost ranker | build_reranked_candidates, build_candidate_ranker_features | initial/refined sets, scores, metadata, candidate library | re-ranked candidate set | candidate library and ranker | fixed 78-D ranker feature contract | candidate construction/order is method behavior; CatBoost loading is asset plumbing |
| 3.7 | Label–Cardinality Joint Probability Estimation | Estimate ten atom probabilities across four cardinalities | src/medical_parsing/tasks/multilabel.py | src/medical_parsing/training/multilabel.py | 40 fitted probability models | build_probability_model_features | score vector, generated/refined/re-ranked sets, metadata | 10 x 4 base probabilities | probability model bundle | fixed 50-D probability feature contract; cardinalities 1,2,3,4 | fitted estimator inference is method behavior; joblib loading is implementation |
| 3.8 | Token-Conditioned Residual Probability Refinement | Add a bounded learned residual using image-token distribution summaries and row features | src/medical_parsing/models/multilabel_head.py | src/medical_parsing/training/multilabel.py | MultiLabelResidualProbabilityHead | forward, predict | [N,256,2560] tokens, normalized row features, base logits | [N,10,4] probabilities | multilabel_residual_head.pt | 128-D token projection; 40 outputs | neural residual is core; device/autocast handling is implementation |
| 3.9 | Cardinality-Aware GFM Decoding | Select the set maximizing the fixed cardinality-aware F1 utility with deterministic ties | src/medical_parsing/tasks/multilabel.py | no separate producer | none | gfm_decode, serialize_effective | 10 x 4 probabilities | canonical atom string | none | fixed F1 utility and atom order | deterministic serialization is output-contract engineering supporting the method |
| 3.10 | Multi-Label Orchestration | Execute proposal, scoring, refinement, re-ranking, probability, residual, and decoding stages | src/medical_parsing/tasks/multilabel.py | external/data-derived feature preparation | MultiLabelResidualProbabilityHead plus fitted estimators | run_multilabel | validated MLC rows | canonical multi-label strings and audit | six MLC assets plus primary adapter | ModelConfig and MultilabelConfig | lifecycle/cleanup/audit fields are implementation details |
| 3.11 | Adaptive Multi-View Construction | Construct original plus aspect-ratio-dependent crops | src/medical_parsing/tasks/regression.py | feature preparation is external | none | build_adaptive_image_views | one loaded image | 4 views for wide/tall; 5 views for near-square | none | crop policy is fixed | view construction is method input behavior; PIL object handling is implementation |
| 3.12 | Visual and Intensity–Edge Geometry Representations | Build 7680-D visual and 960-D intensity/edge descriptors | src/medical_parsing/tasks/regression.py | src/medical_parsing/training/regression.py | none | _extract_visual_features, geometry_one, l2 | adaptive views and one image | normalized visual/geometry representations | raw base model; persisted scalers/PCA in reference/model assets | feature dimensions and PCA=64 | pooling and normalization are method behavior; batching and cache cleanup are implementation |
| 3.13 | Visual Regression Estimator and Generated Numeric Estimator | Produce visual and generated numeric estimates | src/medical_parsing/tasks/regression.py | src/medical_parsing/training/regression.py | fitted visual estimator; external regression adapter | run_regression, _first_number | visual features and regression prompt | visual_regression_estimate and generated_numeric_estimate | regression_visual_model.joblib; regression adapter | 0.5/0.5 base fusion | estimator loading and numeric parsing are supporting implementation |
| 3.14 | Cross-Group Retrieval and Residual Correction | Retrieve same-contract references while excluding the query group; retrieve residuals similarly | src/medical_parsing/tasks/regression.py | src/medical_parsing/training/regression.py | none | retrieve_reg, weighted_median | normalized fused representation, groups, targets/residuals | retrieval_estimate and retrieved_residual_correction | regression_reference.joblib; regression_residuals.npz | retrieval_neighbors=15 | group exclusion, weighted median, and tie order are method behavior |
| 3.15 | Spatial Quantile Refinement Head | Use coordinate-aware tokens and four learned queries to return ordered Q25/Q50/Q75 estimates | src/medical_parsing/models/regression_head.py | src/medical_parsing/training/regression.py | SpatialQuantileRefinementHead | forward | [N,256,2560] tokens and 64-D geometry | ordered lower/median/upper quantile estimates | regression_quantile_head.pt; primary adapter | 128-D projection; 4 queries; Q25/Q50/Q75 | head computation is core; autocast/device lifecycle is implementation |
| 3.16 | Regression Fusion and Canonicalization | Fuse base, retrieval, residual, and upper-quantile estimates and clip to [0,100] | src/medical_parsing/tasks/regression.py | no separate producer | none | fuse_regression_estimates | named numerical estimates | bounded numeric prediction | all four regression assets | RegressionConfig fusion weights | fusion equation and clipping are method/output contract |
| 3.17 | Component-wise Training/Fitting | Fit each learned or data-derived external asset separately | train.py and src/medical_parsing/training/ | n/a | training helper classes | explicit component dispatcher | labeled rows or prepared feature arrays | one named external asset | user-owned labeled data and external base/adapters as applicable | component defaults and fixed schemas | fitting math is method support; CLI and serialization are reproducibility engineering |
| 3.18 | Inference Orchestration and Canonicalization | Validate rows, group task branches, write canonical output, and record audit | src/medical_parsing/inference/pipeline.py; src/medical_parsing/schema.py | n/a | AssetBundle | run_inference, validate_input_rows, validate_output_rows | unlabeled JSONL/JSON with exactly one image per row | canonical JSONL and audit | task-specific assets | default.yaml | input validation, output validation, offline mode, and cleanup are implementation/reproducibility |
| 3.19 | Diagnostic/Local Evaluation | Report local accuracy, set metrics, and regression errors | src/medical_parsing/evaluation/metrics.py | n/a | none | evaluate_rows, evaluate_files | labeled reference and canonical prediction rows | diagnostic metrics JSON | none | none | not the organizer's official scorer |

## Runtime configuration

configs/default.yaml is the only runtime configuration source.

| Key | Default | Consumer | Meaning |
| --- | ---: | --- | --- |
| model.name | google/medgemma-1.5-4b-it | model loading | expected base identity |
| model.image_size | 896 | image preparation | square image preparation size |
| model.feature_batch_size | 8 | raw visual extraction | image-token/view extraction batch |
| model.vocab_chunk_size | 16 | MLC scorer | vocabulary-row chunk size |
| model.max_new_tokens | 256 | generation | greedy generation limit |
| model.seed | 0 | determinism/training helpers | default seed |
| multilabel.candidate_change_threshold | 0.05 | candidate selector | minimum selector change |
| multilabel.max_replacement_candidates | 32 | candidate table | replacement candidate cap |
| multilabel.scoring_row_batch_size | 4 | MLC scorer | row batch for ten singleton answers |
| regression.retrieval_neighbors | 15 | retrieval | cross-group neighbor count |
| regression.generated_fusion_weight | 0.5 | regression | generated/visual base fusion |
| regression.visual_fusion_weight | 0.5 | regression | generated/visual base fusion |
| regression.base_fusion_weight | 0.75 | regression | base/retrieval and corrected/quantile fusion |
| regression.retrieval_fusion_weight | 0.25 | regression | base/retrieval fusion |
| regression.residual_correction_weight | 0.5 | regression | residual correction strength |
| regression.quantile_fusion_weight | 0.25 | regression | upper-quantile fusion |
| regression.quantiles | [0.25, 0.50, 0.75] | quantile training contract | ordered quantile outputs |
| regression.geometry_pca_components | 64 | fitting contract | persisted geometry PCA size |

The task YAML files are reference-only component contracts. They are not
silently interpreted as runtime overrides.

## External asset contract

| Asset | Runtime consumer | Public producer | Status |
| --- | --- | --- | --- |
| classification_route_manifest.json | classification routing | none; external/data-derived curation | external |
| classification_heads.pt | SemanticImageTokenHead ensemble | train.py --component classification-head | covered |
| multilabel_template_map.json | MLC metadata | none; external/data-derived preparation | external |
| multilabel_candidate_library.json | candidate construction | none; external/data-derived preparation | external |
| multilabel_candidate_selector.cbm | candidate refinement | train.py --component multilabel-selector-ranker | covered |
| multilabel_candidate_ranker.cbm | candidate re-ranking | train.py --component multilabel-selector-ranker | covered |
| multilabel_probability_models.joblib | atom/cardinality probabilities | train.py --component multilabel-probability-models | covered |
| multilabel_residual_head.pt | residual probability refinement | train.py --component multilabel-residual-head | covered |
| regression_visual_model.joblib | visual estimate | train.py --component regression-visual-estimator | covered |
| regression_reference.joblib | retrieval representation/table | train.py --component regression-reference | covered |
| regression_residuals.npz | residual retrieval | train.py --component regression-residuals | covered |
| regression_quantile_head.pt | spatial quantile estimate | train.py --component regression-quantile-head | covered |
| MedGemma base | all multimodal branches | external | external |
| primary LoRA adapter | generation, MLC scoring, spatial-token branch | generic API only; no exact final-adapter claim | external |
| regression LoRA adapter | generated numeric estimate | generic API only; no exact final-adapter claim | external |

## Compatibility names

The following old names remain thin aliases to preserve callers and serialized
feature contracts. They are not separate paper modules:

- image_features -> extract_image_tokens
- make_semantic_head -> SemanticImageTokenHead
- load_semantic_heads -> load_semantic_head_ensemble
- resolve_slot -> resolve_classification_slot
- semantic_concept -> map_option_to_semantic_concept
- serialize_concept -> map_semantic_concept_to_option
- make_multilabel_model -> MultiLabelResidualProbabilityHead
- SEMANTIC/PSEUDO -> SEMANTIC_LABEL_ATOMS/AUXILIARY_LABEL_ATOMS
- candidate_table/pnss_features/choose_pnss/ranked_candidates/ranked_features ->
  their formally named candidate helpers
- token_chunk -> vocab_chunk_size
- tile_views -> build_adaptive_image_views
- final_regression_blend -> fuse_regression_estimates
