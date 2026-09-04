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

# Ambarella CVTools runs in the prebuilt Podman container. The setup script can
# load/start/reuse it and write the verified name into env.sh:
#   ./setup_conda_envs.sh --ambarella-only
# Set this manually only when the container was created outside that command.
# The host $HOME is mounted into the container, so the repository remains
# visible there.
export AMBARELLA_CONTAINER=""
# The host-side converter uses this Python only to inspect ONNX/YAML before it
# calls podman exec. It does not run Statlas quantization. The fallback is
# STATLAS_PYTHON, then python3; the selected interpreter needs onnx and PyYAML.
export AMBARELLA_PYTHON="${STATLAS_PYTHON}"
# Bundle directory containing RunContainer.sh and the image loader. By
# default setup searches dependencies/amba, including extracted subfolders.
export AMBARELLA_CONTAINER_DIR="/path/to/quant_folder/dependencies/amba"
# The image archive is loaded only when the expected image is absent. It is
# about 30 GB, so --dry-run never loads it and normal Conda setup does not
# initialize Ambarella unless --ambarella/--ambarella-only is supplied.
export AMBARELLA_IMAGE_TAR="/path/to/quant_folder/dependencies/ambacontainer_2404_cuda12.9_cudnn_sdk_onnx_v3.9.1.0.tar"
# Optional overrides used by the setup script.
# export AMBARELLA_RUN_SCRIPT="${AMBARELLA_CONTAINER_DIR}/RunContainer.sh"
# export AMBARELLA_IMAGE_LOADER="${AMBARELLA_CONTAINER_DIR}/AmbaContainerPreBuildImageLoader.sh"
# export AMBARELLA_IMAGE="ambacontainer_2404_cuda12.9_cudnn/sdk_onnx:3.9.1.0"
# Optional: override the ADK project template or CVTools paths used inside the
# container. The default template is /home/falcon2/my_model_build.
# export AMBARELLA_TEMPLATE_DIR="/home/falcon2/my_model_build"
# export AMBARELLA_ADK_PATH="/opt/cvtools/sample_nn_diag/diags/onnx/yolo_v8_s_ox/../../../../adk"
