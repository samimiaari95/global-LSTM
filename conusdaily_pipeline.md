# CONUSdaily water-table depth evaluation

CONUSdaily compares daily observed and predicted water-table depth (WTD) in metres from 1 March 2022 through 28 February 2024. ERA5-Land defines the evaluation grid, and the ParFlow CONUS2 snapshot supplies static predictors. Run the commands from the repository root on JURECA.

## 1. Activate the environment and locate the sources

```bash
source /p/project1/cslts/miaari1/python_scripts/DailyScriptBox/envs/loadenv.JURECA-DC_JUWELS_2025_GCC-OpenMPI_Std-Py-AI.ini
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
```

The raw HydroData USGS observations are in `inputs/CONUS/observations/daily/`. If the archive needs to be downloaded, install the HydroData client in the activated environment and run `python -m LSTM_model.data.conus_observations download`. The downloader reads `CONUS_HYDRODATA_PIN` or prompts for the registered account PIN.

The retained source bundle in `inputs/CONUSdaily/source/prepared/` covers **955** ERA5-Land cells and **1,229** wells. It contains the aligned 365-day forcing lookback, target dates, and dynamic and static predictors. The source directory also preserves the forcing samples, station mappings, and input exclusion records. The corresponding 100 daily predictions and Slurm logs are in `outputs/CONUSdaily/source/`.

## 2. Prepare gridded daily WTD and model inputs

```bash
python -m LSTM_model.data.conusdaily_evaluation prepare
python -m LSTM_model.data.conus_inputs preflight --root inputs/CONUSdaily
```

Each contributing well needs finite WTD on all 730 target dates. February 29 is excluded. Wells at the same cell have equal weight and the set of wells stays fixed through the period. Preparation retains **660 wells at 554 cells**. It writes aligned input arrays, calendars, the well inventory, and `observations_gridded.csv` to `inputs/CONUSdaily/prepared/`.

## 3. Obtain predictions

```bash
python -m LSTM_model.data.conusdaily_evaluation import-predictions
```

This validates all 100 retained source members and copies the matching columns to `outputs/CONUSdaily/predictions/`. For fresh inference on the daily subset, use the daily launcher.

```bash
bash submit_conusdaily_prediction.sh
```

The script submits one job to JURECA-DC's `dc-gpu` partition. It reserves one node and four GPUs for the job and runs members 000–099 in order. It writes to `outputs/CONUSdaily/predictions_fresh/` and records the full run in `outputs/CONUSdaily/logs/prediction_<jobid>.out` and `.err`. A resubmission skips members that already have both output files. Pass `--predictions outputs/CONUSdaily/predictions_fresh` when evaluating a fresh run.

Direct submission with `sbatch submit_conusdaily_prediction.sh` also works from the repository root and uses Slurm's default `slurm-<jobid>.out` log.

## 4. Evaluate and plot

```bash
python -m LSTM_model.data.conusdaily_evaluation evaluate \
  --output outputs/CONUSdaily/evaluation/run_daily_new
```

Evaluation matches dates and cells exactly and compares daily WTD to the mean of 100 destandardized member predictions. Bias is prediction minus observation. RMSE, MAE and bias use metres. Pearson r and NSE require variable observations. Pearson r also requires variable predictions.

The [existing figure run](outputs/CONUSdaily/evaluation/run_2022_2024_plots/) contains the matched NetCDF series, cell scores and ensemble statistics, metric maps and CDFs, statistics versus metrics, paired scatter plots, and daily series with 5th–95th ensemble percentile bands. Use `--plot-cells 0 10` to limit individual figures. Summary plots cover all cells. Use fresh destinations when rerunning preparation, import or evaluation.
