# CONUSdaily water-table depth evaluation

CONUSdaily compares daily observed and predicted water-table depth (WTD) in metres from 1 March 2022 through 28 February 2024. ERA5-Land defines the evaluation grid, and the ParFlow CONUS2 snapshot supplies static predictors. Run the commands from the repository root on JURECA.

## 1. Activate the environment and locate the sources

```bash
source /p/project1/cslts/miaari1/python_scripts/DailyScriptBox/envs/loadenv.JURECA-DC_JUWELS_2025_GCC-OpenMPI_Std-Py-AI.ini
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
```

The raw HydroData USGS observations are in `inputs/CONUS/observations/daily/`. If the archive needs to be downloaded, install the HydroData client in the activated environment and run `python -m LSTM_model.conusmonthly.conus_observations download`. The downloader reads `CONUS_HYDRODATA_PIN` or prompts for the registered account PIN.

The retained source bundle in `inputs/CONUSdaily/source/prepared/` covers **955** ERA5-Land cells and **1,229** wells. It contains the aligned 365-day forcing lookback, target dates, and dynamic and static predictors. The source directory also preserves the forcing samples, station mappings, and input exclusion records. The corresponding 100 daily predictions and Slurm logs are in `outputs/CONUSdaily/source/`.

## 2. Prepare gridded daily WTD and model inputs

```bash
python -m LSTM_model.conusdaily.conusdaily_evaluation prepare
python -m LSTM_model.conusmonthly.conus_inputs preflight --root inputs/CONUSdaily
```

Each contributing well needs finite WTD on all 730 target dates. February 29 is excluded. Wells at the same cell have equal weight and the set of wells stays fixed through the period. Preparation retains **660 wells at 554 cells**. It writes aligned input arrays, calendars, the well inventory, and `observations_gridded.csv` to `inputs/CONUSdaily/prepared/`.

## 3. Obtain predictions

```bash
python -m LSTM_model.conusdaily.conusdaily_evaluation import-predictions
```

This validates all 100 retained source members and copies the matching columns to `outputs/CONUSdaily/predictions/`. For fresh inference on the daily subset, use the daily launcher.

```bash
bash scripts/conusdaily/submit_conusdaily_prediction.sh
```

The script submits one job to JURECA-DC's `dc-gpu` partition. It reserves one node and four GPUs for the job and runs members 000–099 in order. It writes to `outputs/CONUSdaily/predictions_fresh/` and records the full run in `outputs/CONUSdaily/logs/prediction_<jobid>.out` and `.err`. A resubmission skips members that already have both output files. Pass `--predictions outputs/CONUSdaily/predictions_fresh` when evaluating a fresh run.

Direct submission with `sbatch scripts/conusdaily/submit_conusdaily_prediction.sh` also works from the repository root and uses Slurm's default `slurm-<jobid>.out` log.

## 4. Evaluate and plot

```bash
python -m LSTM_model.conusdaily.conusdaily_evaluation evaluate \
  --output outputs/CONUSdaily/evaluation/run_daily_new
```

Evaluation matches dates and cells exactly and compares daily WTD to the mean of 100 destandardized member predictions. The daily figures use absolute mean bias, calculated as the absolute value of the mean daily prediction-minus-observation error. The CSV also retains signed mean bias. RMSE, MAE and both bias measures use metres.

The daily evaluator uses the original Kling–Gupta efficiency, `KGE = 1 − sqrt((r − 1)² + (α − 1)² + (β − 1)²)`, where `r` is Pearson correlation, `α` is the ratio of predicted to observed standard deviations, and `β` is the ratio of predicted to observed means. KGE is undefined if either series is constant or the observed mean is zero. NSE requires variable observations, and Pearson correlation requires both series to vary. [Gupta et al. (2009)](https://doi.org/10.1016/j.jhydrol.2009.08.003)

The [updated figure run](../outputs/CONUSdaily/evaluation/run_2022_2024_kge/) contains the matched NetCDF series, per-cell scores, maps and CDFs, ensemble statistics versus metrics, and paired scatter plots. Daily time series show the observations and ensemble mean in colour and all 100 member traces in transparent grey.

- The RMSE, MAE and absolute mean bias maps, CDFs and statistics-versus-metrics panels use logarithmic scales. KGE and NSE use symmetric logarithmic scales with an upper limit of 1. Pearson correlation uses a linear scale.
- Ensemble variance, IQR, standard deviation and mean absolute deviation use logarithmic axes in the statistics-versus-metrics panels. Mean pairwise member correlation stays linear. Spread and temporal standard deviation use logarithmic colour scales on the statistics maps.
- Exact zeros remain in the tables and CDF counts. Logarithmic axes and colour scales display positive values. CDFs use all defined scores in their denominator.

Use `--plot-cells 0 10` to limit individual figures. Summary plots cover all cells. Use fresh destinations when rerunning preparation, import or evaluation.
