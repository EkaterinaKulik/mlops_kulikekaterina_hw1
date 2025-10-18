#!/usr/bin/env bash
set -euo pipefail

python3 src/preprocessing.py
python3 src/scorer.py
python3 src/export_outputs.py
