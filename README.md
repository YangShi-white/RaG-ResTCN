# RaG-ResTCN: Raman-Gated Residual Forecasting for Fed-Batch Fermentation

## English

This repository contains the clean review package for the paper project on controlled multi-step forecasting, Raman-assisted residual learning, anomaly diagnostics, and uncertainty evaluation in industrial fed-batch fermentation.

The main model is **RaG-ResTCN**, a target-wise Raman-gated residual temporal convolutional network. The implementation separates known future control inputs from unknown endogenous process states, fits a train-normal controlled ridge baseline, and trains a causal residual learner to model nonlinear corrections. Raman spectra are injected through target-wise gates rather than unconditional concatenation.

### Repository contents

```text
configs/                 Experiment and data configuration files
data/                    Raw-data placement instructions only; raw data are not included
docs/                    Data availability and reproducibility notes
hpc/slurm/               Example SLURM scripts for cluster execution
results/                 Paper table CSVs and figure-source CSVs
scripts/                 Data preparation, model training, baseline, diagnostic, and external Raman scripts
src/fermnftp/            Shared Python package code
```

### Data

The IndPenSim raw benchmark data are not redistributed in this repository. Obtain them from:

Goldrick, Stephen (2019), "Data for: Modern day monitoring and control challenges outlined on an industrial-scale benchmark fermentation process", Mendeley Data, V1, doi: `10.17632/pdnjz7zz5x.1`.

Place the two CSV files under:

```text
data/raw/
```

Expected filenames:

```text
100_Batches_IndPenSim_V3.csv
100_Batches_IndPenSim_Statistics.csv
```

The external Raman validation script expects the Grebe et al. Raman spectral files and reference concentrations dataset associated with doi: `10.1002/bit.70211`. Place that dataset in a local folder and pass it with `--data-root`.

### Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
```

### Minimal reproduction workflow

Run data preparation first:

```bash
python scripts/03_data_audit.py --config configs/data/indpensim_local.json
python scripts/04_preprocess_indpensim.py --config configs/data/preprocess_phase04.json
python scripts/05_build_splits.py --config configs/data/split_phase04.json
python scripts/07_fit_train_normal_stats.py --processed-root data/processed/phase04
```

Run Raman preprocessing and the residual forecasting model:

```bash
python scripts/13_run_phase08_raman_multimodal.py --config configs/model/phase08_raman_multimodal.json
python scripts/17_train_phase10_residual_multimodal.py --config configs/model/phase10_residual_multimodal.json --device auto
```

Run the main evidence-generation and baseline scripts:

```bash
python scripts/25_run_phase15_strict_gap_completion.py --config configs/model/phase15_strict_gap_completion.json
python scripts/26_run_phase16_strong_baselines.py --config configs/model/phase16_strong_baselines.json --device auto --n-jobs 4
python scripts/30_run_phase16_anomaly_only.py --config configs/model/phase16_anomaly_only.json --device cpu --n-jobs 16
```

Run the external Raman support experiment when the external Raman dataset is available:

```bash
python scripts/27_run_phase17_rwth_external_raman.py --config configs/model/phase17_rwth_external_raman.json --data-root /path/to/Raman_spectral_files_and_reference_concentrations --n-jobs 16
```

For cluster execution, edit the paths in `hpc/slurm/*.slurm` and submit the corresponding job.

### Notes

- Raw data are intentionally not included.
- Intermediate checkpoints, logs, local return folders, and LaTeX build files are intentionally not included.
- Paper table CSVs and figure-source CSVs are provided under `results/` for inspection.
- All paths in the public configs are relative, except for user-specified data locations passed at runtime.

---

## 中文

本仓库是论文项目的 clean review 版本，用于编辑和审稿人查看代码、配置和关键结果表。项目研究工业补料分批发酵过程中的受控多步预测、Raman 辅助残差学习、异常诊断和不确定性评估。

核心模型是 **RaG-ResTCN**，即目标级 Raman 门控残差时序卷积网络。它把未来已知控制输入和未来未知过程状态分开处理，先用正常训练批次拟合受控 ridge 基线，再用因果残差网络学习非线性修正。Raman 光谱不是简单拼接到所有目标上，而是通过目标级门控选择性进入预测。

### 仓库内容

```text
configs/                 数据和实验配置
data/                    原始数据放置说明；不包含原始数据
docs/                    数据来源和复现实验说明
hpc/slurm/               超算 SLURM 示例脚本
results/                 论文表格 CSV 和图表源数据 CSV
scripts/                 数据处理、模型训练、基线、诊断和外部 Raman 验证脚本
src/fermnftp/            项目共享 Python 代码
```

### 数据

本仓库不重新分发 IndPenSim 原始数据。原始数据来源为：

Goldrick, Stephen (2019), "Data for: Modern day monitoring and control challenges outlined on an industrial-scale benchmark fermentation process", Mendeley Data, V1, doi: `10.17632/pdnjz7zz5x.1`.

请将两个 CSV 文件放在：

```text
data/raw/
```

期望文件名：

```text
100_Batches_IndPenSim_V3.csv
100_Batches_IndPenSim_Statistics.csv
```

外部 Raman 验证脚本使用 Grebe 等人的 Raman 光谱和参考浓度数据集，DOI 为 `10.1002/bit.70211`。该数据集需要本地放置，并通过 `--data-root` 指定。

### 安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
```

### 最小复现实验流程

先运行数据处理：

```bash
python scripts/03_data_audit.py --config configs/data/indpensim_local.json
python scripts/04_preprocess_indpensim.py --config configs/data/preprocess_phase04.json
python scripts/05_build_splits.py --config configs/data/split_phase04.json
python scripts/07_fit_train_normal_stats.py --processed-root data/processed/phase04
```

再运行 Raman 预处理和残差预测模型：

```bash
python scripts/13_run_phase08_raman_multimodal.py --config configs/model/phase08_raman_multimodal.json
python scripts/17_train_phase10_residual_multimodal.py --config configs/model/phase10_residual_multimodal.json --device auto
```

运行主要证据生成和强基线实验：

```bash
python scripts/25_run_phase15_strict_gap_completion.py --config configs/model/phase15_strict_gap_completion.json
python scripts/26_run_phase16_strong_baselines.py --config configs/model/phase16_strong_baselines.json --device auto --n-jobs 4
python scripts/30_run_phase16_anomaly_only.py --config configs/model/phase16_anomaly_only.json --device cpu --n-jobs 16
```

如果已经准备好外部 Raman 数据集，可以运行：

```bash
python scripts/27_run_phase17_rwth_external_raman.py --config configs/model/phase17_rwth_external_raman.json --data-root /path/to/Raman_spectral_files_and_reference_concentrations --n-jobs 16
```

如果使用超算，请先修改 `hpc/slurm/*.slurm` 中的路径，再提交任务。

### 说明

- 本仓库不包含原始数据。
- 本仓库不包含中间 checkpoint、运行日志、本地回传目录或 LaTeX 编译缓存。
- `results/` 中保留了论文表格 CSV 和图表源数据 CSV，便于核查。
- 公开配置文件使用相对路径；外部数据路径由运行命令指定。
