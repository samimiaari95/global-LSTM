# CONUS coordinate sensitivity

This experiment tests how replacing out-of-range geographical inputs changes the European LSTM ensemble's accuracy and predicted water-table depth (WTD) anomalies in CONUS. It uses the existing [CONUS monthly workflow](conusmonthly_pipeline.md), including the 784 ERA5-Land cells, 984 mapped NWIS candidate wells, January 2020 through December 2024, and the original 365-day lookback. Daily inference uses the 1,825 prepared no-leap target dates. Evaluation uses their 60 monthly aggregates.

## Training and transfer coordinates

The preparation script reads the original European TSMP training arrays from `spatio-temporal-LSTM/inputs/20yrs_ts/ensemble_400px/400px_member_*/`. For each member it checks that the coordinate columns are constant over the original first 5,475 training days and that their means and standard deviations agree with the fitted scalers to float32 precision. The experiment means pool the 400 training columns of each of the 100 members with equal weight. Repeated days would give the same mean. A cell sampled by multiple members contributes once per occurrence.

| Coordinate | European training range | European training mean | CONUS range | CONUS cells below the European minimum |
| --- | ---: | ---: | ---: | ---: |
| Longitude (degrees east) | −44.593864 to 64.964378 | 24.399684 | −122.500000 to −67.300003 | 784 of 784 |
| Latitude (degrees north) | 33.311722 to 71.127647 | 51.193954 | 29.400000 to 48.500000 | 97 of 784 |

No CONUS coordinates exceed the European maxima. The pooled range describes all 100 members together. Individual members use their own training scalers.

| Scenario | Longitude input | Latitude input |
| --- | --- | --- |
| A0 (archived) | Original CONUS | Original CONUS |
| A1 | European training mean | Original CONUS |
| A2 | Original CONUS | European training mean |
| A3 | European training mean | European training mean |

Preparation snapshots the CONUS inputs inside `outputs/coordinate_sensitivity/prepared`. Scenarios share the forcing, calendars, pixel IDs, well mapping and observations. Each scenario has its own raw float32 static feature array. The predictor applies each member's fitted feature means and standard deviations to every lookback sample after the coordinate substitution. It restores WTD units using that member's fitted target mean and standard deviation. The experiment uses the existing LSTM architecture and all 100 pretrained checkpoints.

The manifest records exact means, ranges, input hashes and model and scaler hashes. Preparation verifies every static element against its expected value and tests normalized batches through the existing inference reader for members 000 and 099. Verification checks also run before prediction and evaluation. Maps and location labels use the original CONUS coordinates.

## Run on JURECA-DC

From the repository root, prepare a new experiment and preview the Slurm command. The default preparation has already run for the current experiment.

```bash
bash scripts/coordinate_sensitivity/prepare.sh
bash scripts/coordinate_sensitivity/submit_prediction.sh --dry-run
```

When ready to launch prediction, use the explicit submission switch. The array runs A1, A2 and A3 one scenario at a time, with four GPUs on one `dc-gpu` node per task. Each scenario task evaluates members 000–099 and publishes each member's array and metadata together. A retry skips completed members. Slurm logs are in `outputs/coordinate_sensitivity/logs`. [JURECA batch system](https://apps.fz-juelich.de/jsc/hps/jureca/batchsystem.html)

```bash
bash scripts/coordinate_sensitivity/submit_prediction.sh --submit
```

A0 comes from the existing 100-member archive in `outputs/CONUSmonthly/predictions`. Those predictions used the same ensemble now stored in `LSTM_model/model/ensemble_400px`. The archive metadata retains the previous checkpoint and scaler paths. Validation checks member filenames, metadata, dates, pixel order, array shapes and finite values. It also compares artifact hashes when a historical path is still available. The evaluator reads the archive directly and checks that its prepared inputs match the experiment snapshot. The A0 static array in the prepared run serves as the reference for checking coordinate substitutions.

When the array completes, run the CPU evaluation. Preparation and evaluation refuse to replace an existing result directory. Use `--run PATH` for preparation and evaluation and pass the same path as the second argument to the submission launcher to create a separate run.

```bash
bash scripts/coordinate_sensitivity/evaluate.sh
```

## Evaluation and interpretation

All four scenarios use the CONUSmonthly observation reference. For each of the 784 prediction cells, the eligible NWIS well closest to the ERA5-Land cell centre supplies the full 60-month WTD series. The shared CONUSmonthly selector requires complete monthly data and variation within each calendar-month climatology. It uses Haversine distance and breaks exact ties by site ID. The per-cell metrics and NetCDF record the selected site ID, its distance to the cell centre, and the number of eligible candidate wells.

The evaluator averages daily forecasts within each month. For each member and observation cell, it calculates calendar-month z-scores relative to that series' own 2020–2024 climatology with population standard deviation. The ensemble anomaly is the equal-weight mean of the 100 member anomalies. It uses one cell-month mask requiring finite observed and ensemble anomalies in all four scenarios. For each metric, the boxplots and deltas use the same cells with defined scores in every scenario. KGE follows the existing monthly convention with a bias ratio of one for zero-mean anomalies.

The evaluation writes Pearson correlation, RMSE, MAE, KGE and NSE by cell, along with each metric's paired scenario-minus-A0 change. It also writes the RMSE between the ensemble anomaly time series and A0 on the common mask. The summary gives medians, quartiles, IQR widths and cell counts. Boxplots share the scale within each metric, and the spatial panels share the CONUS extent and a symmetric, zero-centered color scale within each metric. The representative series share axis limits and use the cells nearest the 10th, 50th and 90th empirical percentiles of the A3-versus-A0 prediction anomaly RMSE.

Preflight established that CONUS longitudes are outside the pooled training range at every cell, while most CONUS latitudes lie within it. Use the paired score changes to assess predictive accuracy and the anomaly RMSE and representative traces to assess how much the modeled dynamics change. Small changes in skill can coexist with large changes in the predicted anomalies.

- [Per-cell metrics and changes](../outputs/coordinate_sensitivity/evaluation/cell_metrics_and_changes.csv)
- [Summary table](../outputs/coordinate_sensitivity/evaluation/metric_summary.csv)
- [Figures](../outputs/coordinate_sensitivity/evaluation/figures/)
- [Run results and interpretation](../outputs/coordinate_sensitivity/evaluation/results.md)
