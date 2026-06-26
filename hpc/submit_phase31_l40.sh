#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/logs"

mkdir -p "${LOG_DIR}"
cd "${PROJECT_ROOT}"

sbatch \
  --chdir="${PROJECT_ROOT}" \
  --export=ALL,PHASE31_PROJECT_ROOT="${PROJECT_ROOT}" \
  --output="${LOG_DIR}/phase31_infosci_modern_deep_baselines_%j.out" \
  --error="${LOG_DIR}/phase31_infosci_modern_deep_baselines_%j.err" \
  hpc/run_phase31_l40.sbatch
