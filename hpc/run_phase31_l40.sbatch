#!/bin/bash
#SBATCH --job-name=infosci_p31
#SBATCH --partition=L40
#SBATCH --gres=gpu:l40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --time=168:00:00
#SBATCH --output=logs/phase31_infosci_modern_deep_baselines_%j.out
#SBATCH --error=logs/phase31_infosci_modern_deep_baselines_%j.err

set -euo pipefail

if [[ -n "${PHASE31_PROJECT_ROOT:-}" ]]; then
  PROJECT_ROOT="${PHASE31_PROJECT_ROOT}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/hpc/run_phase31_l40.sbatch" ]]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
cd "${PROJECT_ROOT}"

mkdir -p logs

{
  echo "[phase31] started at $(date '+%F %T')"
  echo "[phase31] host=$(hostname)"
  echo "[phase31] pwd=$(pwd)"
  echo "[phase31] project_root=${PROJECT_ROOT}"
  echo "[phase31] slurm_submit_dir=${SLURM_SUBMIT_DIR:-unset}"
  echo "[phase31] job_id=${SLURM_JOB_ID:-manual}"
  echo "[phase31] cuda visible devices=${CUDA_VISIBLE_DEVICES:-unset}"
} | tee "logs/phase31_infosci_modern_deep_baselines_${SLURM_JOB_ID:-manual}_bootstrap.log"

module load cuda/11.8

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
fi
source activate sy520

python -u scripts/31_run_infosci_modern_deep_baselines.py --config configs/model/phase31_infosci_modern_deep_baselines.json
