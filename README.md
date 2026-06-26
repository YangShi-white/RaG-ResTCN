# RaG-ResTCN

[中文说明](README_zh.md)

RaG-ResTCN is a controlled residual learning framework for multi-horizon forecasting and fault-sensitive monitoring in fed-batch fermentation. It separates known future exogenous controls from unknown future endogenous states, combines a train-normal controlled ridge baseline with a causal residual temporal convolutional network, and supports target-wise Raman residual fusion.

This repository contains source code, experiment scripts, public configurations, and cluster submission examples only. Raw data, processed data, checkpoints, logs, result files, figures, and manuscript files are intentionally excluded.

## Repository layout

```text
configs/data/            Data audit, preprocessing, split, and variable-role configs
configs/model/           Forecasting, Raman, diagnostic, baseline, and ablation configs
docs/                    Data availability and reproducibility notes
hpc/                     SLURM submission examples
scripts/                 Data preparation and experiment entry points
src/fermnftp/            Shared data, metric, model, and plotting utilities
pyproject.toml            Python package metadata
requirements.txt         Runtime dependencies
```

## Requirements

- Python 3.10 or newer
- CUDA-capable PyTorch installation for GPU experiments
- Linux is recommended for the provided SLURM scripts

Create an environment and install the project:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

For GPU execution, install the PyTorch build matching the CUDA driver on the target machine before running the experiments.

## Data

Data are not distributed in this repository.

### IndPenSim

Obtain the IndPenSim benchmark from the original Mendeley Data record:

> Goldrick, Stephen (2019), "Data for: Modern day monitoring and control challenges outlined on an industrial-scale benchmark fermentation process," Mendeley Data, V1. DOI: `10.17632/pdnjz7zz5x.1`.

Create `data/raw/` and place these files in it:

```text
100_Batches_IndPenSim_V3.csv
100_Batches_IndPenSim_Statistics.csv
```

The default public configurations use repository-relative paths. Update `configs/data/indpensim_local.json` only if the files are stored elsewhere.

### External Raman data

The external Raman experiment uses the Raman spectral files and reference concentrations associated with DOI `10.1002/bit.70211`. Keep that dataset outside the repository and provide its location through `--data-root`.

## Data preparation

Run the preparation stages in order:

```bash
python scripts/03_data_audit.py --config configs/data/indpensim_local.json
python scripts/04_preprocess_indpensim.py --config configs/data/preprocess_phase04.json
python scripts/05_build_splits.py --config configs/data/split_phase04.json
python scripts/07_fit_train_normal_stats.py --processed-root data/processed/phase04
```

The split is batch based. Scaling statistics, model-selection quantities, and monitoring thresholds are estimated from the designated training/validation partitions rather than from held-out test batches.

## Main experiments

Raman preprocessing and residual forecasting:

```bash
python scripts/13_run_phase08_raman_multimodal.py \
  --config configs/model/phase08_raman_multimodal.json

python scripts/17_train_phase10_residual_multimodal.py \
  --config configs/model/phase10_residual_multimodal.json \
  --device auto
```

Main component, baseline, anomaly, and external Raman experiments:

```bash
python scripts/25_run_phase15_strict_gap_completion.py \
  --config configs/model/phase15_strict_gap_completion.json

python scripts/26_run_phase16_strong_baselines.py \
  --config configs/model/phase16_strong_baselines.json \
  --device auto --n-jobs 4

python scripts/30_run_phase16_anomaly_only.py \
  --config configs/model/phase16_anomaly_only.json \
  --device cpu --n-jobs 16

python scripts/27_run_phase17_rwth_external_raman.py \
  --config configs/model/phase17_rwth_external_raman.json \
  --data-root /path/to/raman_dataset \
  --n-jobs 16
```

## Modern deep baselines and future-control ablation

The Phase 31 experiment evaluates Direct TCN, DLinear, iTransformer-Lite, and TSMixer over five horizons, four control-input modes, and five random seeds.

Single-process execution:

```bash
python scripts/31_run_infosci_modern_deep_baselines.py \
  --config configs/model/phase31_infosci_modern_deep_baselines.json
```

On a SLURM cluster with up to six L40 GPUs:

```bash
bash hpc/submit_phase31_l40_array6.sh
```

After all array tasks finish, merge the shard metrics:

```bash
python scripts/32_collect_phase31_shards.py
```

Review the partition, CUDA module, environment name, and resource requests in `hpc/` before submission.

## Reproducibility notes

- Experiment behavior is controlled by the JSON files under `configs/`.
- Generated files are written under ignored output directories and should remain outside version control.
- The repository does not include reported numerical results. Reproducing them requires the corresponding datasets and compute environment.
- See [docs/reproducibility.md](docs/reproducibility.md) and [docs/data_availability.md](docs/data_availability.md) for additional details.

## Citation

The manuscript citation will be added after publication. Until then, cite this repository and the original datasets used in the experiment.
