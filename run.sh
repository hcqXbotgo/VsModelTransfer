#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ENV_FILE="${STATLAS_ENV_FILE:-${ROOT}/env.sh}"
# Keep an explicitly supplied Ambarella container name ahead of the local
# env.sh value.  This is useful when switching between already-created
# containers without rewriting the machine-specific file.
CALLER_AMBARELLA_CONTAINER_SET="${AMBARELLA_CONTAINER+x}"
CALLER_AMBARELLA_CONTAINER="${AMBARELLA_CONTAINER-}"
if [[ ! -f "${ENV_FILE}" ]]; then
    echo "Environment config not found: ${ENV_FILE}" >&2
    echo "Run: cp ${ROOT}/env.example.sh ${ROOT}/env.sh" >&2
    echo "Then edit env.sh for this machine." >&2
    exit 2
fi

# shellcheck source=/dev/null
source "${ENV_FILE}"
if [[ -n "${CALLER_AMBARELLA_CONTAINER_SET}" ]]; then
    export AMBARELLA_CONTAINER="${CALLER_AMBARELLA_CONTAINER}"
fi
exec python3 "${ROOT}/run.py" "$@"
