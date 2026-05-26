#!/bin/bash
#SBATCH --job-name=strong_baselines
#SBATCH --partition=intel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=72:00:00
#SBATCH --output=logs/strong_baselines_%j.out
#SBATCH --error=logs/strong_baselines_%j.err

set -euo pipefail

PROJECT_ROOT=/path/to/rag-restcn-fermentation
cd "${PROJECT_ROOT}"
mkdir -p logs

source activate sy520
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

python scripts/26_run_phase16_strong_baselines.py --config configs/model/phase16_strong_baselines.json --device cpu --n-jobs 48
