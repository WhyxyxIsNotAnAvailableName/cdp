#!/usr/bin/env bash
set -euo pipefail

CDP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-${CDP_ROOT}/configs/train.yaml}"
DDP_CONFIG="${DDP_CONFIG:-${CDP_ROOT}/configs/ddp.yaml}"
NUM_PROCESSES="${NUM_PROCESSES:-2}"
export PYTHONPATH="${CDP_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

exec "${PYTHON}" -m accelerate.commands.launch \
    --config_file "${DDP_CONFIG}" \
    --num_processes "${NUM_PROCESSES}" \
    -m cdp.training.train \
    --config_yaml "${CONFIG}" \
    "$@"
