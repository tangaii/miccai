# Data preparation

The repository expects user-provided data.  Official FLARE data and images are
not redistributed here.

The public input schema is JSONL with one object per row:

```json
{
  "uid": "example-0001",
  "task_type": "classification",
  "dataset": "fundus",
  "prompt": "What is the diagnosis? A: normal B: glaucoma",
  "images": ["images/example-0001.jpg"]
}
```

Supported tasks are `classification`, `multi_label_classification`, and
`regression`.  Each row needs a unique `uid`, a dataset/source name, exactly
one local image reference, and a prompt or question.  Inputs must not contain
answer, target, label, reference, or prediction fields.  The validator rejects
multi-image rows because all final task branches consume one image per row.

`scripts/prepare_data.py` is an inference-manifest preparation tool only.  It
normalizes unlabeled records and deliberately rejects training labels.  It is
not a source-data converter and does not produce any learned checkpoint.
Training requires a separate user-owned labeled table/feature cache matching
the fitting function documented in `checkpoints/README.md` and the explicit
component CLI in the root `train.py`.

Store user data outside the repository when its license or size does not
permit redistribution.  A typical local layout is:

```text
data-root/
├── raw/
├── processed/
└── metadata/
```

The smoke-data generator used by tests creates synthetic images only and is
not an official metric or data reproduction.
