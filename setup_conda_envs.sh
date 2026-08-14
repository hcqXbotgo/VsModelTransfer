#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_ROOT="${QUANT_CONDA_ENV_ROOT:-${ROOT}/dependencies/conda-envs}"
STATLAS_ENV="${STATLAS_CONDA_ENV_DIR:-${ENV_ROOT}/statlas}"
RKNN_ENV="${RKNN_CONDA_ENV_DIR:-${ENV_ROOT}/rknn}"
STATLAS_DIR="${ROOT}/dependencies/statlas"
RKNN_DIR="${ROOT}/dependencies/rknn-toolkit2-2.3.2/rknn-toolkit2/packages/x86_64"
ENV_FILE="${STATLAS_ENV_FILE:-${ROOT}/env.sh}"
DRY_RUN=0
INSTALL_STATLAS=1
INSTALL_RKNN=1

usage() {
    cat <<'EOF'
Usage: ./setup_conda_envs.sh [options]

Create repository-local Conda environments for the Statlas and RKNN toolchains.

Options:
  --statlas-only       Only create/update the Statlas Python 3.8 environment
  --rknn-only          Only create/update the RKNN Toolkit2 Python 3.9 environment
  --env-root PATH      Environment parent directory (default: dependencies/conda-envs)
  --dry-run            Print commands without creating environments or env.sh
  -h, --help           Show this help

Environment overrides:
  CONDA_EXE             Conda executable
  QUANT_CONDA_ENV_ROOT  Same purpose as --env-root
  STATLAS_CONDA_ENV_DIR Override the Statlas environment prefix
  RKNN_CONDA_ENV_DIR    Override the RKNN environment prefix
  STATLAS_ENV_FILE      Generated runtime config (default: env.sh)
EOF
}

while (($#)); do
    case "$1" in
        --statlas-only)
            INSTALL_RKNN=0
            ;;
        --rknn-only)
            INSTALL_STATLAS=0
            ;;
        --env-root)
            [[ $# -ge 2 ]] || { echo "--env-root requires a path" >&2; exit 2; }
            ENV_ROOT="$(realpath -m "$2")"
            STATLAS_ENV="${ENV_ROOT}/statlas"
            RKNN_ENV="${ENV_ROOT}/rknn"
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

if [[ "$(uname -m)" != "x86_64" ]]; then
    echo "This setup uses the bundled x86_64 wheels; current architecture: $(uname -m)" >&2
    exit 1
fi

CONDA_BIN="${CONDA_EXE:-$(command -v conda || true)}"
if [[ -z "${CONDA_BIN}" || ! -x "${CONDA_BIN}" ]]; then
    echo "conda was not found. Install Conda or set CONDA_EXE=/path/to/conda." >&2
    exit 1
fi

run() {
    printf '+'
    printf ' %q' "$@"
    printf '\n'
    if ((DRY_RUN == 0)); then
        "$@"
    fi
}

require_file() {
    [[ -f "$1" ]] || { echo "Required file not found: $1" >&2; exit 1; }
}

ensure_env() {
    local prefix="$1"
    local version="$2"
    if [[ ! -x "${prefix}/bin/python" ]]; then
        run "${CONDA_BIN}" create --yes --prefix "${prefix}" "python=${version}" pip
    else
        echo "Using existing environment: ${prefix}"
    fi
    run "${prefix}/bin/python" -m pip install --timeout 300 --retries 5 \
        --upgrade 'pip<25'
}

install_statlas() {
    local wheel="${STATLAS_DIR}/statlas_quant-0.0.1+cpu-cp38-cp38-linux_x86_64.whl"
    require_file "${wheel}"
    require_file "${STATLAS_DIR}/StatlasCompile"
    ensure_env "${STATLAS_ENV}" 3.8
    run "${STATLAS_ENV}/bin/python" -m pip install --timeout 300 --retries 5 \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        'torch==2.1.0+cpu' 'torchvision==0.16.0+cpu'
    run "${STATLAS_ENV}/bin/python" -m pip install --timeout 300 --retries 5 \
        "${wheel}"
    if ((DRY_RUN == 0)); then
        "${STATLAS_ENV}/bin/python" -c 'import onnx, torch, yaml'
        [[ -x "${STATLAS_ENV}/bin/StatlasQuant" ]] || {
            echo "StatlasQuant entry point was not installed" >&2
            exit 1
        }
    fi
}

install_rknn() {
    local requirements="${RKNN_DIR}/requirements_cp39-2.3.2.txt"
    local wheel="${RKNN_DIR}/rknn_toolkit2-2.3.2-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
    require_file "${requirements}"
    require_file "${wheel}"
    ensure_env "${RKNN_ENV}" 3.9
    run "${RKNN_ENV}/bin/python" -m pip install --timeout 300 --retries 5 \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        'torch==2.4.0+cpu' 'torchvision==0.19.0+cpu'
    run "${RKNN_ENV}/bin/python" -m pip install --timeout 300 --retries 5 \
        -r "${requirements}"
    run "${RKNN_ENV}/bin/python" -m pip install --timeout 300 --retries 5 \
        'onnx==1.16.2' PyYAML Pillow pycocotools
    run "${RKNN_ENV}/bin/python" -m pip install --timeout 300 --retries 5 \
        "${wheel}"
    if ((DRY_RUN == 0)); then
        "${RKNN_ENV}/bin/python" -c \
            'from rknn.api import RKNN; import PIL, pycocotools, torchvision; print("RKNN Toolkit2 evaluation environment OK")'
    fi
}

write_env_file() {
    local begin='# BEGIN setup_conda_envs.sh managed block'
    local end='# END setup_conda_envs.sh managed block'
    local temp
    temp="$(mktemp "${ENV_FILE}.tmp.XXXXXX")"
    if [[ -f "${ENV_FILE}" ]]; then
        awk -v begin="${begin}" -v end="${end}" '
            $0 == begin { skip = 1; next }
            $0 == end { skip = 0; next }
            !skip { print }
        ' "${ENV_FILE}" > "${temp}"
    else
        printf '%s\n' '#!/usr/bin/env bash' '# Local machine configuration. This file is ignored by Git.' > "${temp}"
    fi
    {
        printf '\n%s\n' "${begin}"
        if ((INSTALL_STATLAS == 1)) || [[ -x "${STATLAS_ENV}/bin/StatlasQuant" ]]; then
            printf 'export STATLAS_CONDA_ENV_DIR=%q\n' "${STATLAS_ENV}"
            printf 'export STATLAS_PYTHON=%q\n' "${STATLAS_ENV}/bin/python"
            printf 'export STATLAS_QUANT=%q\n' "${STATLAS_ENV}/bin/StatlasQuant"
            printf 'export STATLAS_COMPILE_DIR=%q\n' "${STATLAS_DIR}"
        fi
        if ((INSTALL_RKNN == 1)) || [[ -x "${RKNN_ENV}/bin/python" ]]; then
            printf 'export RKNN_CONDA_ENV_DIR=%q\n' "${RKNN_ENV}"
            printf 'export RKNN_PYTHON=%q\n' "${RKNN_ENV}/bin/python"
        fi
        printf '%s\n' "${end}"
    } >> "${temp}"
    mv "${temp}" "${ENV_FILE}"
    chmod 600 "${ENV_FILE}"
}

((INSTALL_STATLAS == 0)) || install_statlas
((INSTALL_RKNN == 0)) || install_rknn

if ((DRY_RUN == 0)); then
    write_env_file
    echo "Environment configuration updated: ${ENV_FILE}"
    echo "Run a conversion with: ./run.sh demo_v5 compile --platform rk3576"
else
    echo "+ update ${ENV_FILE} with Statlas and RKNN environment paths"
fi
