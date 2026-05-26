#!/bin/bash
#SBATCH --job-name=anomaly_baselines
#SBATCH --partition=intel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=72:00:00
#SBATCH --output=logs/anomaly_baselines_%j.out
#SBATCH --error=logs/anomaly_baselines_%j.err

set -euo pipefail

PROJECT_ROOT=/path/to/rag-restcn-fermentation
cd "${PROJECT_ROOT}"
mkdir -p logs

source activate sy520
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

python scripts/30_run_phase16_anomaly_only.py --config configs/model/phase16_anomaly_only.json --device cpu --n-jobs 48
