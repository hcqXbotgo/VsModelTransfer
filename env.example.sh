#!/usr/bin/env bash
# Copy this file to env.sh and edit only the values for the local machine.
# env.sh is intentionally ignored by Git.

# These repository-local environments can be created automatically with:
# ./setup_conda_envs.sh
export STATLAS_CONDA_ENV_DIR="/path/to/quant_folder/dependencies/conda-envs/statlas"

# Normally these two are derived from STATLAS_CONDA_ENV_DIR.
export STATLAS_PYTHON="${STATLAS_CONDA_ENV_DIR}/bin/python"
export STATLAS_QUANT="${STATLAS_CONDA_ENV_DIR}/bin/StatlasQuant"

# Directory containing StatlasCompile, lib/, and its runtime dependencies.
export STATLAS_COMPILE_DIR="/path/to/quant_folder/dependencies/statlas"

# Python environment containing Rockchip RKNN Toolkit2 (rknn.api).
# Required only for: ./run.sh <mode> compile --platform rk3576
export RKNN_CONDA_ENV_DIR="/path/to/quant_folder/dependencies/conda-envs/rknn"
export RKNN_PYTHON="${RKNN_CONDA_ENV_DIR}/bin/python"
