#!/bin/bash
#SBATCH --job-name=rag_restcn
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=168:00:00
#SBATCH --output=logs/rag_restcn_%j.out
#SBATCH --error=logs/rag_restcn_%j.err

set -euo pipefail

PROJECT_ROOT=/path/to/rag-restcn-fermentation
cd "${PROJECT_ROOT}"
mkdir -p logs

module load cuda/11.8
source activate sy520
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

python scripts/13_run_phase08_raman_multimodal.py --config configs/model/phase08_raman_multimodal.json
python scripts/17_train_phase10_residual_multimodal.py --config configs/model/phase10_residual_multimodal.json --device auto
python scripts/25_run_phase15_strict_gap_completion.py --config configs/model/phase15_strict_gap_completion.json
