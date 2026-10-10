#!/usr/bin/env bash
set -euo pipefail

CDP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-${CDP_ROOT}/configs/train.yaml}"
export PYTHONPATH="${CDP_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

exec "${PYTHON}" -m cdp.preprocessing.run_preprocessing \
    --config_yaml "${CONFIG}" \
    --gpus "${GROUNDING_GPUS:-0}" \
    --workers-per-gpu "${WORKERS_PER_GPU:-1}" \
    "$@"
