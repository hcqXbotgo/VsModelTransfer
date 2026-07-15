#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ENV_FILE="${STATLAS_ENV_FILE:-${ROOT}/env.sh}"
if [[ ! -f "${ENV_FILE}" ]]; then
    echo "Environment config not found: ${ENV_FILE}" >&2
    echo "Run: cp ${ROOT}/env.example.sh ${ROOT}/env.sh" >&2
    echo "Then edit env.sh for this machine." >&2
    exit 2
fi

# shellcheck source=/dev/null
source "${ENV_FILE}"
exec python3 "${ROOT}/run.py" "$@"
