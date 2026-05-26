#!/bin/bash
#SBATCH --job-name=external_raman
#SBATCH --partition=intel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=72:00:00
#SBATCH --output=logs/external_raman_%j.out
#SBATCH --error=logs/external_raman_%j.err

set -euo pipefail

PROJECT_ROOT=/path/to/rag-restcn-fermentation
RWTH_DATA_ROOT=/path/to/Raman_spectral_files_and_reference_concentrations
cd "${PROJECT_ROOT}"
mkdir -p logs

source activate sy520
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

python scripts/27_run_phase17_rwth_external_raman.py --config configs/model/phase17_rwth_external_raman.json --data-root "${RWTH_DATA_ROOT}" --n-jobs 48
