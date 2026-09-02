# R3-MedGemma Method-Code Map

This document is the canonical paper-to-code map for the frozen public
R3-MedGemma implementation. It has exactly four paper-level method modules. Detailed
multi-label and regression operations are internal stages of sections 3.3 and
3.4; validation, asset loading, training, and serialization remain
implementation or reproducibility support.

## Top-level module map

| Paper section | Formal module name | Primary inference code | Primary neural code | Output |
| --- | --- | --- | --- | --- |
| 3.1 | Unified Framework and Shared MedGemma Backbone | `src/medical_parsing/inference/pipeline.py`, `src/medical_parsing/schema.py` | `src/medical_parsing/models/backbone.py` | projected image tokens, generated text, decoder states |
| 3.2 | Task-Routed Semantic–Generative Classification | `src/medical_parsing/tasks/classification.py` | `src/medical_parsing/models/classification_head.py` | one legal option letter |
| 3.3 | Structured Evidence-Refined Multi-Label Prediction | `src/medical_parsing/tasks/multilabel.py` | `src/medical_parsing/models/multilabel_head.py` | canonical set of legal label atoms |
| 3.4 | Multi-View Retrieval–Quantile Regression | `src/medical_parsing/tasks/regression.py` | `src/medical_parsing/models/regression_head.py` | bounded numeric value in [0, 100] |

## Internal stage map

| Paper section | Module | Internal stage | Code | Function/class | Frozen role |
| --- | --- | --- | --- | --- | --- |
| 3.1 | Shared backbone | Input and image preparation | `schema.py` | `validate_input_rows`, `single_image_ref`, `load_image`, `prepared_image` | exactly one local image; EXIF transpose, RGB conversion, deterministic 896 x 896 resize |
| 3.1 | Shared backbone | Sequential model states | `models/backbone.py` | `load_raw_bundle`, `load_adapter_bundle` | raw base, primary adapter, and regression adapter are loaded as separate states; adapters are not stacked |
| 3.1 | Shared backbone | Projected image tokens | `models/backbone.py` | `extract_image_tokens` | `get_image_features(pixel_values=...)` returns `[N, 256, 2560]` tokens |
| 3.1 | Shared backbone | Decoder access and generation | `models/backbone.py` | `decoder_hidden_states`, `generated_text` | greedy generation and decoder hidden states share the MedGemma representation |
| 3.2 | Classification | Semantic image-token head | `models/classification_head.py` | `SemanticImageTokenHead`, `load_semantic_head_ensemble` | LayerNorm, 2560-to-128 projection, distribution summary, source-specific heads, three-fold probability mean |
| 3.2 | Classification | Semantic concept and option mapping | `models/classification_head.py` | `resolve_classification_slot`, `map_option_to_semantic_concept`, `map_semantic_concept_to_option` | ontology concepts are mapped back to each row's legal option letters |
| 3.2 | Classification | Task routing | `tasks/classification.py` | `route_for_row`, `run_classification` | `semantic_head`, `direct_prompt_generation`, and `instructional_generation_fallback`; semantic inference is active for bone marrow, fundus, and IUGC |
| 3.3 | MLC | Initial generative proposal | `tasks/multilabel.py` | `generated_text`, `parse_multilabel` | generate and parse the initial legal atom set S0 |
| 3.3 | MLC | Teacher-forced singleton evidence | `tasks/multilabel.py` | `score_teacher_forced_streaming`, `score_hidden_streaming` | score ten parser-native singleton answers under the actual chat template |
| 3.3 | MLC | Memory-efficient vocabulary scoring | `tasks/multilabel.py` | `score_hidden_streaming` | row batch 4 and vocabulary-row chunks of 16; equivalent to full-logit reference scoring |
| 3.3 | MLC | Thresholded candidate refinement | `tasks/multilabel.py` | `build_initial_candidate_table`, `build_candidate_selector_features`, `select_refined_candidate` | KEEP/ADD/DROP/nearby replacement candidates with change threshold 0.05 |
| 3.3 | MLC | Listwise re-ranking | `tasks/multilabel.py` | `build_reranked_candidates`, `build_candidate_ranker_features` | rank the refined candidate set and alternatives using the fixed feature contract |
| 3.3 | MLC | Tooth-position correction | `tasks/multilabel.py` | `apply_tooth_position_aware_correction` | for FDI tooth position 8, a non-KEEP choice is restored to candidate 0, the PNSS-refined set; other rows retain the ranked choice |
| 3.3 | MLC | Label-cardinality probabilities | `tasks/multilabel.py` | `build_probability_model_features`, `run_multilabel` | ten atoms by four cardinalities, yielding a 10 x 4 probability matrix |
| 3.3 | MLC | Token-conditioned residual | `models/multilabel_head.py` | `MultiLabelResidualProbabilityHead` | 256 x 2560 tokens are projected to 128-D; quantile, mean, and standard-deviation summaries produce a bounded residual |
| 3.3 | MLC | Cardinality-aware decoding | `tasks/multilabel.py` | `gfm_decode`, `serialize_effective` | maximize the fixed F1 utility over prediction cardinalities 1 through 10 with deterministic ties |
| 3.4 | Regression | Adaptive views | `tasks/regression.py` | `build_adaptive_image_views` | wide/tall: original plus three crops; near-square: original plus four quadrant crops |
| 3.4 | Regression | Visual and geometry representations | `tasks/regression.py` | `_extract_visual_features`, `geometry_one`, `l2` | 7,680-D visual representation and 960-D intensity/edge descriptor |
| 3.4 | Regression | Visual and generated estimates | `tasks/regression.py` | `run_regression`, `_first_number` | fitted visual estimator and generated numeric estimate are fused with equal weights |
| 3.4 | Regression | Cross-group retrieval | `tasks/regression.py` | `retrieve_reg`, `weighted_median` | K=15 non-identical image-group neighbors with inverse-distance weighted median |
| 3.4 | Regression | Spatial quantile refinement | `models/regression_head.py` | `SpatialQuantileRefinementHead` | 128-D token projection, 2-D coordinates, four learned queries, ordered Q25/Q50/Q75 |
| 3.4 | Regression | Final fusion | `tasks/regression.py` | `fuse_regression_estimates` | residual correction and upper-quantile fusion, clipped to [0, 100] |

## Runtime configuration

`configs/default.yaml` is the only runtime configuration source. Task YAML files
are reference-only component contracts.

The public implementation and a final deployment may schedule vocabulary rows
differently for memory-bounded scoring. These schedules are algebraically
equivalent and preserve the same candidate-score definition; this is not a
prediction-semantic change.

| Key | Default | Consumer | Meaning |
| --- | ---: | --- | --- |
| `model.name` | `google/medgemma-1.5-4b-it` | model loading | expected base identity |
| `model.image_size` | 896 | image preparation | square preparation size |
| `model.feature_batch_size` | 8 | visual extraction | extraction batch size |
| `model.vocab_chunk_size` | 16 | MLC scorer | vocabulary-row chunk size |
| `model.max_new_tokens` | 256 | generation | greedy generation limit |
| `model.seed` | 0 | determinism | default seed |
| `multilabel.candidate_change_threshold` | 0.05 | candidate selector | minimum change score |
| `multilabel.max_replacement_candidates` | 32 | candidate table | replacement cap |
| `multilabel.scoring_row_batch_size` | 4 | MLC scorer | rows per ten-answer scoring batch |
| `regression.retrieval_neighbors` | 15 | retrieval | cross-group neighbor count |
| `regression.generated_fusion_weight` | 0.5 | regression | generated/visual base fusion |
| `regression.visual_fusion_weight` | 0.5 | regression | generated/visual base fusion |
| `regression.base_fusion_weight` | 0.75 | regression | base/retrieval and corrected/quantile fusion |
| `regression.retrieval_fusion_weight` | 0.25 | regression | base/retrieval fusion |
| `regression.residual_correction_weight` | 0.5 | regression | residual correction strength |
| `regression.quantile_fusion_weight` | 0.25 | regression | upper-quantile fusion |
| `regression.quantiles` | `[0.25, 0.50, 0.75]` | quantile head | ordered quantile outputs |
| `regression.geometry_pca_components` | 64 | fitting contract | persisted geometry PCA size |

## Implementation and reproducibility support

These surfaces support the four modules but are not additional paper-level
method sections:

- `src/medical_parsing/config.py` resolves runtime defaults and external asset names.
- `src/medical_parsing/inference/pipeline.py` dispatches validated rows, writes canonical JSONL, and records audits.
- `src/medical_parsing/evaluation/metrics.py` reports local diagnostic metrics; it is not the organizer scorer.
- `src/medical_parsing/training/` and `train.py` fit separately named external components. Generic adapter training does not claim to reproduce the challenge adapters exactly.
- `tests/` verifies contracts, tensor shapes, deterministic ties, targeted tooth-position behavior, and streaming equivalence.

## External asset contract

| Asset | Runtime consumer | Public producer | Status |
| --- | --- | --- | --- |
| `classification_route_manifest.json` | classification routing | external/data-derived curation | external |
| `classification_heads.pt` | three-fold `SemanticImageTokenHead` ensemble | `train.py --component classification-head` | covered |
| `multilabel_template_map.json` | MLC metadata | external/data-derived preparation | external |
| `multilabel_candidate_library.json` | candidate construction | external/data-derived preparation | external |
| `multilabel_candidate_selector.cbm` | candidate refinement | `train.py --component multilabel-selector-ranker` | covered |
| `multilabel_candidate_ranker.cbm` | candidate re-ranking | `train.py --component multilabel-selector-ranker` | covered |
| `multilabel_probability_models.joblib` | atom/cardinality probabilities | `train.py --component multilabel-probability-models` | covered |
| `multilabel_residual_head.pt` | residual refinement | `train.py --component multilabel-residual-head` | covered |
| `regression_visual_model.joblib` | visual estimate | `train.py --component regression-visual-estimator` | covered |
| `regression_reference.joblib` | retrieval representation/table | `train.py --component regression-reference` | covered |
| `regression_residuals.npz` | residual retrieval | `train.py --component regression-residuals` | covered |
| `regression_quantile_head.pt` | spatial quantile estimate | `train.py --component regression-quantile-head` | covered |
| MedGemma base | all multimodal branches | external | external |
| primary LoRA adapter | generation, MLC scoring, spatial-token branch | generic API only; exact final adapter training is not claimed | external |
| regression LoRA adapter | generated numeric estimate | generic API only; exact final adapter training is not claimed | external |

## Compatibility names

The following old names remain thin aliases for callers and serialized feature
contracts. They are not separate paper modules:

- `image_features` -> `extract_image_tokens`
- `make_semantic_head` -> `SemanticImageTokenHead`
- `load_semantic_heads` -> `load_semantic_head_ensemble`
- `resolve_slot` -> `resolve_classification_slot`
- `semantic_concept` -> `map_option_to_semantic_concept`
- `serialize_concept` -> `map_semantic_concept_to_option`
- `make_multilabel_model` -> `MultiLabelResidualProbabilityHead`
- `SEMANTIC`/`PSEUDO` -> `SEMANTIC_LABEL_ATOMS`/`AUXILIARY_LABEL_ATOMS`
- `candidate_table`/`pnss_features`/`choose_pnss`/`ranked_candidates`/`ranked_features` -> their formally named helpers
- `token_chunk` -> `vocab_chunk_size`
- `tile_views` -> `build_adaptive_image_views`
- `final_regression_blend` -> `fuse_regression_estimates`
