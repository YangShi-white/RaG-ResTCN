# RaG-ResTCN

[English README](README.md)

RaG-ResTCN 是面向补料分批发酵多步预测与故障敏感监测的受控残差学习框架。该框架明确区分未来已知的外生控制量与未来未知的内生过程状态，先利用正常训练批次拟合受控 Ridge 基线，再通过因果残差时序卷积网络学习非线性修正，并支持目标级 Raman 残差融合。

本仓库只包含源代码、实验程序、公开配置和超算提交示例。原始数据、处理后数据、模型权重、日志、结果文件、图表和论文文件均不上传。

## 仓库结构

```text
configs/data/            数据审计、预处理、划分和变量角色配置
configs/model/           预测、Raman、诊断、基线和消融实验配置
docs/                    数据可用性与复现说明
hpc/                     SLURM 超算提交示例
scripts/                 数据处理与实验入口程序
src/fermnftp/            公共数据、指标、模型和绘图工具
pyproject.toml            Python 项目元数据
requirements.txt         运行依赖
```

## 环境要求

- Python 3.10 或更高版本
- GPU 实验需要与 CUDA 驱动匹配的 PyTorch
- `hpc/` 中的 SLURM 脚本建议在 Linux 集群上使用

创建环境并安装项目：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

在 GPU 环境中，请先根据目标机器的 CUDA 版本安装对应的 PyTorch 构建版本。

## 数据准备

本仓库不分发任何数据。

### IndPenSim

请从原始 Mendeley Data 页面获取 IndPenSim 数据：

> Goldrick, Stephen (2019), "Data for: Modern day monitoring and control challenges outlined on an industrial-scale benchmark fermentation process," Mendeley Data, V1. DOI：`10.17632/pdnjz7zz5x.1`。

创建 `data/raw/`，并放入以下文件：

```text
100_Batches_IndPenSim_V3.csv
100_Batches_IndPenSim_Statistics.csv
```

公开配置默认使用仓库相对路径。如果数据位于其他目录，请修改 `configs/data/indpensim_local.json`。

### 外部 Raman 数据

外部 Raman 实验使用 DOI `10.1002/bit.70211` 对应的 Raman 光谱和参考浓度数据。请将数据保存在仓库之外，并在运行时通过 `--data-root` 指定目录。

## 数据处理流程

按顺序运行：

```bash
python scripts/03_data_audit.py --config configs/data/indpensim_local.json
python scripts/04_preprocess_indpensim.py --config configs/data/preprocess_phase04.json
python scripts/05_build_splits.py --config configs/data/split_phase04.json
python scripts/07_fit_train_normal_stats.py --processed-root data/processed/phase04
```

数据采用批次级划分。缩放统计量、模型选择参数和监测阈值只使用指定的训练集或验证集估计，不使用保留测试批次。

## 主要实验

Raman 预处理与残差预测：

```bash
python scripts/13_run_phase08_raman_multimodal.py \
  --config configs/model/phase08_raman_multimodal.json

python scripts/17_train_phase10_residual_multimodal.py \
  --config configs/model/phase10_residual_multimodal.json \
  --device auto
```

主要组件、强基线、异常检测和外部 Raman 实验：

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

## 现代深度基线与未来控制量消融

Phase 31 实验包含 Direct TCN、DLinear、iTransformer-Lite 和 TSMixer，覆盖 5 个预测步长、4 种控制量输入方式和 5 个随机种子。

单进程运行：

```bash
python scripts/31_run_infosci_modern_deep_baselines.py \
  --config configs/model/phase31_infosci_modern_deep_baselines.json
```

在最多使用 6 张 L40 GPU 的 SLURM 集群上运行：

```bash
bash hpc/submit_phase31_l40_array6.sh
```

所有数组任务完成后合并分片结果：

```bash
python scripts/32_collect_phase31_shards.py
```

提交任务前，请根据实际集群检查 `hpc/` 中的分区、CUDA 模块、Conda 环境名称和资源参数。

## 复现说明

- 所有实验行为由 `configs/` 中的 JSON 文件控制。
- 运行生成的文件应保存在 Git 忽略的输出目录中，不应提交到版本库。
- 本仓库不包含论文数值结果；完整复现需要自行准备相应数据与计算环境。
- 更多信息见 [docs/reproducibility.md](docs/reproducibility.md) 和 [docs/data_availability.md](docs/data_availability.md)。

## 引用

论文正式发表后将在此补充引用信息。在此之前，请引用本仓库以及实验所使用的原始数据集。
