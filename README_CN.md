# RaG-ResTCN

这是论文中 IndPenSim 核心实验对应的代码包。仓库保留受控残差预测、可选 Raman 残差融合、预测基线、正常数据校准的故障监测、不确定性分析、未来控制消融和统一 IndPenSim 协议相关代码。

## 数据集

本仓库不包含原始数据。

核心实验使用 IndPenSim 基准数据集：

Goldrick, Stephen (2019), "Data for: Modern day monitoring and control challenges outlined on an industrial-scale benchmark fermentation process", Mendeley Data, V1, doi: `10.17632/pdnjz7zz5x.1`。

请将原始 CSV 文件放入：

```text
data/raw/
```

期望文件名：

```text
100_Batches_IndPenSim_V3.csv
100_Batches_IndPenSim_Statistics.csv
```

预处理后的数据生成到：

```text
data/processed/
```

## 项目结构

```text
configs/
  data/                     IndPenSim 数据、预处理、划分和变量角色配置
  model/                    核心模型、基线、监测、消融和统一协议配置

data/
  raw/                      用户自行放置的 IndPenSim 原始 CSV，不纳入版本管理
  processed/                预处理数组、scaler 和 split 文件，不纳入版本管理

scripts/
  03_data_audit.py          检查原始 IndPenSim 文件
  04_preprocess_indpensim.py
                            预处理 IndPenSim 过程变量和 Raman 字段
  05_build_splits.py        构建批次级训练/验证/测试划分
  07_fit_train_normal_stats.py
                            拟合 train-normal scaler
  13_run_phase08_raman_multimodal.py
                            Raman 预处理和 PCA 特征生成
  17_train_phase10_residual_multimodal.py
                            受控 ridge anchor 与残差 TCN/Raman 变体
  25_run_phase15_strict_gap_completion.py
                            主要残差预测、监测、不确定性和 bootstrap 分析
  26_run_phase16_strong_baselines.py
                            XGBoost、随机森林、高斯过程、PatchTST-style 和异常检测基线
  30_run_phase16_anomaly_only.py
                            仅运行异常检测基线的包装脚本
  31_run_infosci_modern_deep_baselines.py
                            Direct TCN、TSMixer、iTransformer-Lite 和 DLinear 未来控制消融
  32_collect_phase31_shards.py
                            direct control ablation 分片结果汇总
  33_prepare_unified_v2_protocol.py
                            生成统一 IndPenSim 协议的重复划分审计文件
  34_run_unified_v2_rag_restcn.py
                            统一残差/Raman 协议运行脚本
  35_collect_unified_v2_results.py
                            统一协议结果汇总脚本

src/fermnftp/
  data.py                   共享数据读取和窗口构建
  metrics.py                回归指标和 CSV 写出
  models.py                 共享神经预测模块
  plot_style.py             共享绘图风格工具

requirements.txt            Python 依赖
pyproject.toml              包元数据
```

## 未包含内容

本仓库不包含原始数据、生成结果、结果表、图表、论文文件、集群提交文件或外部验证实验文件。
