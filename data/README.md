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
`regression`.  Each row needs a unique `uid`, a dataset/source name, one or
more local image references, and a prompt or question.  Inputs must not
contain answer, target, label, or prediction fields.

For training, use `scripts/prepare_data.py` to convert source JSON records to a
normalized JSONL manifest.  Store user data outside the repository when its
license or size does not permit redistribution.  A typical local layout is:

```text
data-root/
├── raw/
├── processed/
└── metadata/
```

The smoke-data generator used by tests creates synthetic images only and is
not an official metric or data reproduction.

