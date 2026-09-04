#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${STATLAS_ENV_FILE:-${ROOT}/env.sh}"

# Values exported by the caller take precedence over the local env file.  Keep
# the presence bit as well as the value so an intentional empty override is
# distinguishable from an unset variable.
_CALLER_AMBARELLA_CONTAINER_SET="${AMBARELLA_CONTAINER+x}"
_CALLER_AMBARELLA_CONTAINER="${AMBARELLA_CONTAINER-}"
_CALLER_AMBARELLA_DIR_SET="${AMBARELLA_CONTAINER_DIR+x}"
_CALLER_AMBARELLA_DIR="${AMBARELLA_CONTAINER_DIR-}"
_CALLER_AMBARELLA_RUN_SET="${AMBARELLA_RUN_SCRIPT+x}"
_CALLER_AMBARELLA_RUN="${AMBARELLA_RUN_SCRIPT-}"
_CALLER_AMBARELLA_LOADER_SET="${AMBARELLA_IMAGE_LOADER+x}"
_CALLER_AMBARELLA_LOADER="${AMBARELLA_IMAGE_LOADER-}"
_CALLER_AMBARELLA_TAR_SET="${AMBARELLA_IMAGE_TAR+x}"
_CALLER_AMBARELLA_TAR="${AMBARELLA_IMAGE_TAR-}"
_CALLER_AMBARELLA_IMAGE_SET="${AMBARELLA_IMAGE+x}"
_CALLER_AMBARELLA_IMAGE="${AMBARELLA_IMAGE-}"

# Reuse values from the local environment file when this script is invoked
# directly (rather than after ``source env.sh``).  env.sh is a local, user-
# controlled file and contains only shell exports/functions for this project.
if [[ -f "${ENV_FILE}" ]]; then
    # Older local env.sh files may reference optional variables before they
    # define them.  Allow those files to be read, then restore nounset for the
    # setup logic below.
    set +u
    # shellcheck disable=SC1090
    source "${ENV_FILE}"
    set -u
fi

[[ -z "${_CALLER_AMBARELLA_CONTAINER_SET}" ]] || AMBARELLA_CONTAINER="${_CALLER_AMBARELLA_CONTAINER}"
[[ -z "${_CALLER_AMBARELLA_DIR_SET}" ]] || AMBARELLA_CONTAINER_DIR="${_CALLER_AMBARELLA_DIR}"
[[ -z "${_CALLER_AMBARELLA_RUN_SET}" ]] || AMBARELLA_RUN_SCRIPT="${_CALLER_AMBARELLA_RUN}"
[[ -z "${_CALLER_AMBARELLA_LOADER_SET}" ]] || AMBARELLA_IMAGE_LOADER="${_CALLER_AMBARELLA_LOADER}"
[[ -z "${_CALLER_AMBARELLA_TAR_SET}" ]] || AMBARELLA_IMAGE_TAR="${_CALLER_AMBARELLA_TAR}"
[[ -z "${_CALLER_AMBARELLA_IMAGE_SET}" ]] || AMBARELLA_IMAGE="${_CALLER_AMBARELLA_IMAGE}"

# When only the bundle directory is overridden, derive its companion scripts
# and archive unless the caller supplied those paths explicitly.
if [[ -n "${_CALLER_AMBARELLA_DIR_SET}" ]]; then
    [[ -n "${_CALLER_AMBARELLA_RUN_SET}" ]] || \
        AMBARELLA_RUN_SCRIPT="${AMBARELLA_CONTAINER_DIR}/RunContainer.sh"
    [[ -n "${_CALLER_AMBARELLA_LOADER_SET}" ]] || \
        AMBARELLA_IMAGE_LOADER="${AMBARELLA_CONTAINER_DIR}/AmbaContainerPreBuildImageLoader.sh"
    [[ -n "${_CALLER_AMBARELLA_TAR_SET}" ]] || \
        AMBARELLA_IMAGE_TAR="${AMBARELLA_CONTAINER_DIR}/ambacontainer_2404_cuda12.9_cudnn_sdk_onnx_v3.9.1.0.tar"
fi

ENV_ROOT="${QUANT_CONDA_ENV_ROOT:-${ROOT}/dependencies/conda-envs}"
STATLAS_ENV="${STATLAS_CONDA_ENV_DIR:-${ENV_ROOT}/statlas}"
RKNN_ENV="${RKNN_CONDA_ENV_DIR:-${ENV_ROOT}/rknn}"
STATLAS_DIR="${ROOT}/dependencies/statlas"
RKNN_DIR="${ROOT}/dependencies/rknn-toolkit2-2.3.2/rknn-toolkit2/packages/x86_64"

# The Ambarella SDK is distributed as a Podman image.  Keep the default bundle
# inside this repository so setup is portable.  An existing dependencies/amba
# directory is used directly; otherwise setup can create it from a release
# archive placed in dependencies/.
AMBARELLA_DEFAULT_DIR="${ROOT}/dependencies/amba"
AMBARELLA_LEGACY_DIR="/home/falcon2/docker/20260612_AmbaContainer_PreBuildImage_v3.9.1.0.2228_ubu2404_onnx"
if [[ -z "${_CALLER_AMBARELLA_DIR_SET}" &&
      "${AMBARELLA_CONTAINER_DIR:-}" == "${AMBARELLA_LEGACY_DIR}" ]]; then
    unset AMBARELLA_CONTAINER_DIR AMBARELLA_RUN_SCRIPT AMBARELLA_IMAGE_LOADER
fi
if [[ -z "${_CALLER_AMBARELLA_TAR_SET}" &&
      "${AMBARELLA_IMAGE_TAR:-}" == "${AMBARELLA_LEGACY_DIR}"/* ]]; then
    unset AMBARELLA_IMAGE_TAR
fi
AMBARELLA_CONTAINER_DIR="${AMBARELLA_CONTAINER_DIR:-${AMBARELLA_DEFAULT_DIR}}"

# Resolve a package extracted below dependencies/amba.  The normal automatic
# extraction puts RunContainer.sh directly in dependencies/amba, while the
# nested lookup also handles an archive that retains its release directory.
if [[ ! -x "${AMBARELLA_CONTAINER_DIR}/RunContainer.sh" &&
      -d "${AMBARELLA_CONTAINER_DIR}" ]]; then
    _amba_run="$(find "${AMBARELLA_CONTAINER_DIR}" -maxdepth 3 -type f \
        -name RunContainer.sh -print -quit 2>/dev/null || true)"
    if [[ -n "${_amba_run}" ]]; then
        AMBARELLA_CONTAINER_DIR="$(dirname "${_amba_run}")"
    fi
    unset _amba_run
fi
AMBARELLA_RUN_SCRIPT="${AMBARELLA_RUN_SCRIPT:-${AMBARELLA_CONTAINER_DIR}/RunContainer.sh}"
AMBARELLA_IMAGE_LOADER="${AMBARELLA_IMAGE_LOADER:-${AMBARELLA_CONTAINER_DIR}/AmbaContainerPreBuildImageLoader.sh}"
if [[ -z "${AMBARELLA_IMAGE_TAR:-}" ]]; then
    _amba_tar="$(find "${AMBARELLA_CONTAINER_DIR}" -maxdepth 2 -type f \
        -name 'ambacontainer*.tar' -print -quit 2>/dev/null || true)"
    AMBARELLA_IMAGE_TAR="${_amba_tar:-${AMBARELLA_CONTAINER_DIR}/ambacontainer_2404_cuda12.9_cudnn_sdk_onnx_v3.9.1.0.tar}"
    unset _amba_tar
fi
AMBARELLA_IMAGE="${AMBARELLA_IMAGE:-ambacontainer_2404_cuda12.9_cudnn/sdk_onnx:3.9.1.0}"
AMBARELLA_PYTHON="${AMBARELLA_PYTHON:-${STATLAS_PYTHON:-python3}}"
AMBARELLA_CONTAINER="${AMBARELLA_CONTAINER:-}"
AMBARELLA_CONTAINER="$(printf '%s' "${AMBARELLA_CONTAINER}" | tr -d '\r\n' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"

DRY_RUN=0
INSTALL_STATLAS=1
INSTALL_RKNN=1
INSTALL_AMBARELLA="${SETUP_AMBARELLA:-0}"
AMBARELLA_IMAGE_TAR_CLI=0
[[ -z "${_CALLER_AMBARELLA_TAR_SET}" ]] || AMBARELLA_IMAGE_TAR_CLI=1

usage() {
    cat <<'EOF'
Usage: ./setup_conda_envs.sh [options]

Create repository-local Conda environments for the Statlas and RKNN toolchains.
Optionally load/start the Ambarella CVTools Podman container and record its
container name in env.sh.

Options:
  --statlas-only       Only create/update the Statlas Python 3.8 environment
  --rknn-only          Only create/update the RKNN Toolkit2 Python 3.9 environment
  --ambarella          Load (if needed) and start/reuse the Ambarella CVTools container
  --ambarella-only     Only initialize the Ambarella CVTools container
  --no-ambarella       Do not initialize the Ambarella container
  --ambarella-dir PATH CVTools container bundle directory
  --ambarella-image-tar PATH
                       Container image archive used when the image is absent
  --env-root PATH      Environment parent directory (default: dependencies/conda-envs)
  --dry-run            Print commands without creating environments or env.sh
  -h, --help           Show this help

Environment overrides:
  CONDA_EXE             Conda executable
  QUANT_CONDA_ENV_ROOT  Same purpose as --env-root
  STATLAS_CONDA_ENV_DIR Override the Statlas environment prefix
  RKNN_CONDA_ENV_DIR    Override the RKNN environment prefix
  STATLAS_ENV_FILE      Generated runtime config (default: env.sh)
  SETUP_AMBARELLA=1     Enable Ambarella setup without a command-line flag
  AMBARELLA_CONTAINER_DIR, AMBARELLA_RUN_SCRIPT,
  AMBARELLA_IMAGE_LOADER, AMBARELLA_IMAGE_TAR, AMBARELLA_IMAGE,
  AMBARELLA_PYTHON
EOF
}

while (($#)); do
    case "$1" in
        --statlas-only)
            INSTALL_RKNN=0
            INSTALL_AMBARELLA=0
            ;;
        --rknn-only)
            INSTALL_STATLAS=0
            INSTALL_AMBARELLA=0
            ;;
        --ambarella|--with-ambarella)
            INSTALL_AMBARELLA=1
            ;;
        --ambarella-only)
            INSTALL_STATLAS=0
            INSTALL_RKNN=0
            INSTALL_AMBARELLA=1
            ;;
        --no-ambarella)
            INSTALL_AMBARELLA=0
            ;;
        --ambarella-dir)
            [[ $# -ge 2 ]] || { echo "--ambarella-dir requires a path" >&2; exit 2; }
            AMBARELLA_CONTAINER_DIR="$(realpath -m "$2")"
            AMBARELLA_RUN_SCRIPT="${AMBARELLA_CONTAINER_DIR}/RunContainer.sh"
            AMBARELLA_IMAGE_LOADER="${AMBARELLA_CONTAINER_DIR}/AmbaContainerPreBuildImageLoader.sh"
            if ((AMBARELLA_IMAGE_TAR_CLI == 0)); then
                AMBARELLA_IMAGE_TAR="${AMBARELLA_CONTAINER_DIR}/ambacontainer_2404_cuda12.9_cudnn_sdk_onnx_v3.9.1.0.tar"
            fi
            shift
            ;;
        --ambarella-image-tar)
            [[ $# -ge 2 ]] || { echo "--ambarella-image-tar requires a path" >&2; exit 2; }
            AMBARELLA_IMAGE_TAR="$(realpath -m "$2")"
            AMBARELLA_IMAGE_TAR_CLI=1
            shift
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
if ((INSTALL_STATLAS == 1 || INSTALL_RKNN == 1)) &&
   [[ -z "${CONDA_BIN}" || ! -x "${CONDA_BIN}" ]]; then
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

prepare_ambarella_bundle() {
    # The downloadable SDK package is a bzip2-compressed release archive.
    # It contains RunContainer.sh and the inner Podman image archive.  Expand
    # it into the repository-local bundle directory once, so users only need
    # to copy the vendor-provided .tar.bz2 into dependencies/.
    local bundle_root="${ROOT}/dependencies/amba"
    local archive
    if [[ -x "${bundle_root}/RunContainer.sh" ]]; then
        return 0
    fi
    archive="$(find "${ROOT}/dependencies" -maxdepth 1 -type f \
        \( -iname '*.tar.bz2' -o -iname '*.tbz2' \) \
        -print -quit 2>/dev/null || true)"
    if [[ -z "${archive}" ]]; then
        return 0
    fi
    echo "Extracting Ambarella SDK bundle: ${archive}"
    mkdir -p "${bundle_root}"
    tar -xjf "${archive}" -C "${bundle_root}" --strip-components=1
    chmod +x "${bundle_root}/RunContainer.sh" \
        "${bundle_root}/AmbaContainerPreBuildImageLoader.sh" 2>/dev/null || true
    if [[ ! -x "${bundle_root}/RunContainer.sh" ]]; then
        echo "Ambarella archive was extracted, but RunContainer.sh was not found under ${bundle_root}." >&2
        exit 1
    fi
    echo "Ambarella SDK bundle ready: ${bundle_root}"
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

ambarella_container_state() {
    podman container inspect --format '{{.State.Status}}' "$1" 2>/dev/null || true
}

ambarella_container_exists() {
    podman container exists "$1" >/dev/null 2>&1
}

ambarella_container_running() {
    [[ "$(ambarella_container_state "$1")" == "running" ]]
}

use_ambarella_container() {
    local candidate="$1"
    if ambarella_container_running "${candidate}"; then
        AMBARELLA_CONTAINER="${candidate}"
        export AMBARELLA_CONTAINER
        echo "Using running Ambarella container: ${AMBARELLA_CONTAINER}"
        return 0
    fi
    if ambarella_container_exists "${candidate}"; then
        echo "Starting existing Ambarella container: ${candidate}"
        run podman start "${candidate}"
        if ambarella_container_running "${candidate}"; then
            AMBARELLA_CONTAINER="${candidate}"
            export AMBARELLA_CONTAINER
            return 0
        fi
        echo "Ambarella container did not reach running state: ${candidate}" >&2
    fi
    return 1
}

ensure_ambarella() {
    if ((DRY_RUN == 1)); then
        echo "Ambarella container setup (dry-run):"
        echo "  bundle: ${AMBARELLA_CONTAINER_DIR}"
        echo "  image:  ${AMBARELLA_IMAGE}"
        if [[ -n "${AMBARELLA_CONTAINER}" ]]; then
            echo "  reuse:  ${AMBARELLA_CONTAINER}"
        else
            echo "  would load ${AMBARELLA_IMAGE_TAR} if the image is absent"
            echo "  would run ${AMBARELLA_RUN_SCRIPT} ${ROOT} /home/falcon2"
        fi
        return 0
    fi

    command -v podman >/dev/null 2>&1 || {
        echo "podman was not found; install Podman or omit --ambarella." >&2
        exit 1
    }
    if [[ -z "${_CALLER_AMBARELLA_DIR_SET}" &&
          "${AMBARELLA_CONTAINER_DIR}" == "${AMBARELLA_DEFAULT_DIR}" ]]; then
        prepare_ambarella_bundle
        if [[ -x "${AMBARELLA_DEFAULT_DIR}/RunContainer.sh" ]]; then
            AMBARELLA_RUN_SCRIPT="${AMBARELLA_DEFAULT_DIR}/RunContainer.sh"
            AMBARELLA_IMAGE_LOADER="${AMBARELLA_DEFAULT_DIR}/AmbaContainerPreBuildImageLoader.sh"
            if [[ -z "${_CALLER_AMBARELLA_TAR_SET}" ]]; then
                _amba_tar="$(find "${AMBARELLA_DEFAULT_DIR}" -maxdepth 2 -type f \
                    -name 'ambacontainer*.tar' -print -quit 2>/dev/null || true)"
                AMBARELLA_IMAGE_TAR="${_amba_tar:-${AMBARELLA_DEFAULT_DIR}/ambacontainer_2404_cuda12.9_cudnn_sdk_onnx_v3.9.1.0.tar}"
                unset _amba_tar
            fi
        fi
    fi
    # Prefer the value already exported by env.sh, then the launcher's saved
    # name.  A stopped container is restarted instead of creating a duplicate.
    if [[ -n "${AMBARELLA_CONTAINER}" ]] && \
       use_ambarella_container "${AMBARELLA_CONTAINER}"; then
        return 0
    fi
    if [[ -s "${HOME}/ContainerName.log" ]]; then
        local saved
        saved="$(tr -d '\r\n' < "${HOME}/ContainerName.log" |
            sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
        if [[ -n "${saved}" ]] && use_ambarella_container "${saved}"; then
            return 0
        fi
    fi

    # Reuse the newest container made from the expected image when possible.
    local existing
    existing="$(podman ps -a --filter "ancestor=${AMBARELLA_IMAGE}" \
        --format '{{.Names}}' 2>/dev/null | sed -n '1p' || true)"
    if [[ -n "${existing}" ]] && use_ambarella_container "${existing}"; then
        return 0
    fi

    if ! podman image exists "${AMBARELLA_IMAGE}"; then
        [[ -x "${AMBARELLA_IMAGE_LOADER}" ]] || {
            echo "Ambarella image loader not found or not executable: ${AMBARELLA_IMAGE_LOADER}" >&2
            exit 1
        }
        [[ -f "${AMBARELLA_IMAGE_TAR}" ]] || {
            echo "Ambarella image archive not found: ${AMBARELLA_IMAGE_TAR}" >&2
            echo "Set AMBARELLA_IMAGE_TAR or use --ambarella-image-tar PATH." >&2
            exit 1
        }
        echo "Ambarella image ${AMBARELLA_IMAGE} is absent; loading archive..."
        run "${AMBARELLA_IMAGE_LOADER}" "${AMBARELLA_IMAGE_TAR}"
        podman image exists "${AMBARELLA_IMAGE}" || {
            echo "Ambarella image loader completed but image is still unavailable: ${AMBARELLA_IMAGE}" >&2
            exit 1
        }
    fi

    # The launcher is needed only when no existing container can be reused.
    [[ -x "${AMBARELLA_RUN_SCRIPT}" ]] || {
        echo "Ambarella RunContainer.sh not found or not executable: ${AMBARELLA_RUN_SCRIPT}" >&2
        exit 1
    }

    local launch_log launch_status launched
    launch_log="$(mktemp "${ROOT}/.ambarella-container.XXXXXX")"
    echo "+ (cd ${AMBARELLA_CONTAINER_DIR} && ${AMBARELLA_RUN_SCRIPT} ${ROOT} /home/falcon2)"
    set +e
    (
        cd "${AMBARELLA_CONTAINER_DIR}" &&
        "${AMBARELLA_RUN_SCRIPT}" "${ROOT}" "/home/falcon2"
    ) 2>&1 | tee "${launch_log}"
    launch_status="${PIPESTATUS[0]}"
    set -e
    if [[ "${launch_status}" -ne 0 ]]; then
        rm -f "${launch_log}"
        echo "Ambarella container startup failed with exit code ${launch_status}." >&2
        exit "${launch_status}"
    fi

    launched="$(sed -n 's/^Please memorise your container: //p' "${launch_log}" |
        tail -n 1 | tr -d '\r' |
        sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
    if [[ -z "${launched}" && -s "${HOME}/ContainerName.log" ]]; then
        launched="$(tr -d '\r\n' < "${HOME}/ContainerName.log" |
            sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
    fi
    rm -f "${launch_log}"
    if [[ -z "${launched}" ]] || ! use_ambarella_container "${launched}"; then
        echo "RunContainer.sh completed but no running container name was found." >&2
        exit 1
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
        printf 'export AMBARELLA_CONTAINER=%q\n' "${AMBARELLA_CONTAINER:-}"
        printf 'export AMBARELLA_CONTAINER_DIR=%q\n' "${AMBARELLA_CONTAINER_DIR}"
        printf 'export AMBARELLA_RUN_SCRIPT=%q\n' "${AMBARELLA_RUN_SCRIPT}"
        printf 'export AMBARELLA_IMAGE_LOADER=%q\n' "${AMBARELLA_IMAGE_LOADER}"
        printf 'export AMBARELLA_IMAGE_TAR=%q\n' "${AMBARELLA_IMAGE_TAR}"
        printf 'export AMBARELLA_IMAGE=%q\n' "${AMBARELLA_IMAGE}"
        printf 'export AMBARELLA_PYTHON=%q\n' "${AMBARELLA_PYTHON}"
        printf '%s\n' "${end}"
    } >> "${temp}"
    mv "${temp}" "${ENV_FILE}"
    chmod 600 "${ENV_FILE}"
}

((INSTALL_STATLAS == 0)) || install_statlas
((INSTALL_RKNN == 0)) || install_rknn
((INSTALL_AMBARELLA == 0)) || ensure_ambarella

if ((DRY_RUN == 0)); then
    write_env_file
    echo "Environment configuration updated: ${ENV_FILE}"
    echo "Reload it in the current shell with: source ${ENV_FILE}"
    if ((INSTALL_AMBARELLA == 1)); then
        echo "Run a conversion with: ./run.sh basketball compile --platform ambarella"
    else
        echo "Run a conversion with: ./run.sh demo_v5 compile --platform rk3576"
    fi
else
    echo "+ update ${ENV_FILE} with Statlas, RKNN, and Ambarella settings"
fi
