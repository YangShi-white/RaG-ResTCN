#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/logs"

mkdir -p "${LOG_DIR}"
cd "${PROJECT_ROOT}"

sbatch \
  --array=0-5 \
  --chdir="${PROJECT_ROOT}" \
  --export=ALL,PHASE31_PROJECT_ROOT="${PROJECT_ROOT}",PHASE31_SHARD_COUNT=6 \
  --output="${LOG_DIR}/phase31_infosci_modern_deep_baselines_%A_%a.out" \
  --error="${LOG_DIR}/phase31_infosci_modern_deep_baselines_%A_%a.err" \
  hpc/run_phase31_l40.sbatch
