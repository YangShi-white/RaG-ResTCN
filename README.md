# RaG-ResTCN

Code package for the IndPenSim-based experiments in the manuscript. The repository contains the core implementation for control-conditioned residual forecasting, optional Raman residual fusion, forecasting baselines, normal-data-calibrated fault monitoring, uncertainty analysis, future-control ablation, and the unified IndPenSim protocol.

## Dataset

Raw data are not included.

The core experiments use the IndPenSim benchmark:

Goldrick, Stephen (2019), "Data for: Modern day monitoring and control challenges outlined on an industrial-scale benchmark fermentation process", Mendeley Data, V1, doi: `10.17632/pdnjz7zz5x.1`.

Place the raw CSV files in:

```text
data/raw/
```

Expected filenames:

```text
100_Batches_IndPenSim_V3.csv
100_Batches_IndPenSim_Statistics.csv
```

Generated processed data should be written under:

```text
data/processed/
```

## Project Structure

```text
configs/
  data/                     IndPenSim data, preprocessing, split, and variable-role configs
  model/                    Core model, baseline, monitoring, ablation, and unified-protocol configs

data/
  raw/                      User-provided IndPenSim raw CSV files, not tracked
  processed/                Generated preprocessed arrays, scalers, and split files, not tracked

scripts/
  03_data_audit.py          Check raw IndPenSim files
  04_preprocess_indpensim.py
                            Preprocess IndPenSim process and Raman fields
  05_build_splits.py        Build batch-level train/validation/test splits
  07_fit_train_normal_stats.py
                            Fit train-normal scalers
  13_run_phase08_raman_multimodal.py
                            Raman preprocessing and PCA feature generation
  17_train_phase10_residual_multimodal.py
                            Controlled ridge anchors and residual TCN/Raman variants
  25_run_phase15_strict_gap_completion.py
                            Main residual forecasting, monitoring, uncertainty, and bootstrap analyses
  26_run_phase16_strong_baselines.py
                            XGBoost, random forest, Gaussian process, PatchTST-style, and anomaly baselines
  30_run_phase16_anomaly_only.py
                            Anomaly-baseline-only wrapper
  31_run_infosci_modern_deep_baselines.py
                            Direct TCN, TSMixer, iTransformer-Lite, and DLinear future-control ablation
  32_collect_phase31_shards.py
                            Collector for sharded direct-ablation outputs
  33_prepare_unified_v2_protocol.py
                            Prepare repeated split audits for the unified IndPenSim protocol
  34_run_unified_v2_rag_restcn.py
                            Unified residual/Raman protocol runner
  35_collect_unified_v2_results.py
                            Unified-protocol result collector

src/fermnftp/
  data.py                   Shared data loading and window construction
  metrics.py                Regression metrics and CSV writing
  models.py                 Shared neural forecasting modules
  plot_style.py             Shared plotting style utilities

requirements.txt            Python dependencies
pyproject.toml              Package metadata
```

## Not Included

This repository does not include raw datasets, generated outputs, result tables, figures, manuscript files, cluster submission files, or external validation experiment files.
