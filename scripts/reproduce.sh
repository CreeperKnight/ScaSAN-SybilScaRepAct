#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" -m pip install -e .
"${PYTHON_BIN}" -m scasan.cli download --output data/raw
"${PYTHON_BIN}" -m scasan.cli prepare \
  --links data/raw/facebook-links.txt.gz \
  --walls data/raw/facebook-wall.txt.gz \
  --output data/processed/facebook
"${PYTHON_BIN}" -m scasan.cli run \
  --config configs/paper.yaml \
  --data data/processed/facebook \
  --output results/paper
