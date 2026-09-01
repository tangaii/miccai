# External checkpoint bundle

Trained assets are intentionally not distributed with this initial release.
Place a user-trained or otherwise authorized bundle in a directory passed with
`--checkpoint-dir`.  The runtime expects these descriptive filenames:

| File | Consumer | Produced by |
| --- | --- | --- |
| `classification_route_manifest.json` | classification routing | data preparation or route curation |
| `classification_heads.pt` | semantic classification branch | `train.py --task classification` |
| `multilabel_template_map.json` | MLC prompt metadata | `train.py --task multilabel` or data preparation |
| `multilabel_candidate_library.json` | structured MLC candidates | `train.py --task multilabel` |
| `multilabel_candidate_selector.cbm` | candidate selector | `train.py --task multilabel` |
| `multilabel_candidate_ranker.cbm` | candidate ranker | `train.py --task multilabel` |
| `multilabel_probability_models.joblib` | atom/cardinality priors | `train.py --task multilabel` |
| `multilabel_residual_head.pt` | token-conditioned residual head | `train.py --task multilabel` |
| `regression_visual_model.joblib` | visual regression estimator | `train.py --task regression` |
| `regression_reference.joblib` | source feature/reference table | `train.py --task regression` |
| `regression_residuals.npz` | cross-fitted residual correction | `train.py --task regression` |
| `regression_quantile_head.pt` | spatial quantile head | `train.py --task regression` |

The base model and two PEFT adapters are also external inputs:

```text
--base BASE_MODEL_DIR
--adapter OFFICIAL_ADAPTER_DIR
--reg-adapter REGRESSION_ADAPTER_DIR
```

The base model must identify as `google/medgemma-1.5-4b-it`; each adapter must
be a single PEFT LoRA adapter trained for that same base.  The runtime loads
the adapters sequentially and never combines adapter weights.

Do not commit trained weights, fitted estimators, private retrieval tables, or
input-specific feature caches.  Keep them in an external directory ignored by
Git.
