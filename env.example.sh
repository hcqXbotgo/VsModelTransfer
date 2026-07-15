#!/usr/bin/env bash
# Copy this file to env.sh and edit only the values for the local machine.
# env.sh is intentionally ignored by Git.

# Conda installation and environment containing StatlasQuant dependencies.
export CONDA_ROOT="/path/to/anaconda3"
export CONDA_ENV_NAME="Falcon2VsSocQuantEnv"
export STATLAS_CONDA_ENV_DIR="${CONDA_ROOT}/envs/${CONDA_ENV_NAME}"

# Normally these two are derived from STATLAS_CONDA_ENV_DIR.
export STATLAS_PYTHON="${STATLAS_CONDA_ENV_DIR}/bin/python"
export STATLAS_QUANT="${STATLAS_CONDA_ENV_DIR}/bin/StatlasQuant"

# Directory containing StatlasCompile, lib/, and its runtime dependencies.
export STATLAS_COMPILE_DIR="/path/to/VS859_ED_release/tools/NPU/statlas"
