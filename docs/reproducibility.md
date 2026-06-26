# Reproducibility Notes

## Environment

Install dependencies with:

```bash
pip install -r requirements.txt
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
```

For GPU runs, install a PyTorch build compatible with the local CUDA environment.

## Core workflow

1. Place raw IndPenSim CSV files in `data/raw/`.
2. Run data audit, preprocessing, split construction, and train-normal statistics.
3. Run Raman preprocessing with `scripts/13_run_phase08_raman_multimodal.py`.
4. Train the residual forecasting model with `scripts/17_train_phase10_residual_multimodal.py`.
5. Run strong baselines and diagnostics with `scripts/26_run_phase16_strong_baselines.py` and `scripts/30_run_phase16_anomaly_only.py`.
6. Optionally run external Raman validation with `scripts/27_run_phase17_rwth_external_raman.py`.

The scripts write generated artifacts under `outputs/`, which is intentionally ignored by Git.

## Generated results

No numerical results, trained weights, predictions, tables, or figures are tracked in this repository. The experiment scripts generate their artifacts locally under `outputs/` or another configured output directory. These directories are ignored by Git.
