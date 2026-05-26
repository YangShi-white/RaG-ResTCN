# Data

Raw data are not included in this repository.

## IndPenSim benchmark

Download the raw benchmark data from:

Goldrick, Stephen (2019), "Data for: Modern day monitoring and control challenges outlined on an industrial-scale benchmark fermentation process", Mendeley Data, V1, doi: `10.17632/pdnjz7zz5x.1`.

Place the following files in `data/raw/`:

```text
100_Batches_IndPenSim_V3.csv
100_Batches_IndPenSim_Statistics.csv
```

## External Raman validation data

The external Raman component-level validation uses the Raman spectral files and reference concentrations dataset associated with doi: `10.1002/bit.70211`.

Keep this dataset outside the repository or under a local ignored directory, then pass its path to:

```bash
python scripts/27_run_phase17_rwth_external_raman.py --data-root /path/to/Raman_spectral_files_and_reference_concentrations
```
