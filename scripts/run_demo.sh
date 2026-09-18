#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python src/main.py --folder sample_data --output outputs
