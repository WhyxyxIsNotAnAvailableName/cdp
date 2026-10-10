#!/usr/bin/env bash

set -Eeuo pipefail

CDP_ROOT="${CDP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONDA_ROOT="${CONDA_ROOT:-${HOME}/miniconda3}"
ENV_NAME="${GROUNDING_ENV_NAME:-cdp_grounded_sam2}"
PROXY_URL="${CDP_PROXY_URL:-http://127.0.0.1:12345}"

on_error() {
    local exit_code=$?
    echo
    echo "[ERROR] Environment setup failed at line ${BASH_LINENO[0]} (exit ${exit_code})." >&2
    exit "${exit_code}"
}
trap on_error ERR

if [[ ! -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
    echo "[ERROR] Conda initialization script not found under ${CONDA_ROOT}." >&2
    exit 1
fi

echo "[INFO] CDP root:          ${CDP_ROOT}"
echo "[INFO] Conda environment: ${ENV_NAME}"
echo "[INFO] Proxy:             ${PROXY_URL}"

unset ALL_PROXY all_proxy || true
export HTTP_PROXY="${PROXY_URL}"
export HTTPS_PROXY="${PROXY_URL}"
export http_proxy="${PROXY_URL}"
export https_proxy="${PROXY_URL}"
export NO_PROXY="127.0.0.1,localhost"
export no_proxy="127.0.0.1,localhost"
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_DEFAULT_TIMEOUT=180
export HF_HUB_DISABLE_XET=1

# shellcheck disable=SC1091
source "${CONDA_ROOT}/etc/profile.d/conda.sh"

if [[ ! -x "${CONDA_ROOT}/envs/${ENV_NAME}/bin/python" ]]; then
    echo "[STEP 1/4] Creating conda environment ${ENV_NAME}..."
    conda create -n "${ENV_NAME}" python=3.10 pip -y
else
    echo "[STEP 1/4] Reusing existing conda environment ${ENV_NAME}."
fi

conda activate "${ENV_NAME}"

echo "[STEP 2/4] Installing packaging tools..."
python -m pip install wheel "setuptools==75.8.2"

echo "[STEP 3/4] Installing PyTorch 2.3.1 with CUDA 12.1 runtime..."
python -m pip install \
    filelock \
    typing-extensions \
    sympy \
    networkx \
    jinja2 \
    fsspec

python -m pip install \
    "torch==2.3.1" \
    "torchvision==0.18.1" \
    --index-url https://download.pytorch.org/whl/cu121

echo "[STEP 4/4] Installing preprocessing dependencies..."
python -m pip install -e "${CDP_ROOT}[preprocess]"

echo "[SUCCESS] CDP preprocessing environment is ready."
echo "  conda activate ${ENV_NAME}"
