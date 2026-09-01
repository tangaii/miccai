#!/usr/bin/env bash
set -euo pipefail
python train.py --task multilabel "$@"
