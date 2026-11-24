# CONUSmonthly water-table depth anomaly evaluation

CONUSmonthly evaluates monthly standardized water-table depth (WTD) anomalies from **January 2020 through December 2024** on the native 0.1° ERA5-Land grid. Run the commands from the repository root on JURECA.

## 1. Activate the environment and select observations

```bash
source /p/project1/cslts/miaari1/python_scripts/DailyScriptBox/envs/loadenv.JURECA-DC_JUWELS_2025_GCC-OpenMPI_Std-Py-AI.ini
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
```

Install optional download clients with the activated interpreter if needed.

```bash
conus_dependencies="$(mktemp -d /tmp/conusmonthly-deps.XXXXXX)"
python -m pip install --target "$conus_dependencies" \
  'numpy<2' 'pandas<3' 'xarray==2024.11.0' 'scipy<1.16' \
  'hf_hydrodata==1.4.17' 'cdsapi==0.7.7' 'zarr==2.18.7' 'aiohttp>=3.10'
export PYTHONPATH="$conus_dependencies${PYTHONPATH:+:$PYTHONPATH}"
```

HydroData reads the registered account PIN from `CONUS_HYDRODATA_PIN` or prompts for it. Configure CDS via `~/.cdsapirc` or set `CDSAPI_RC` to an authorized configuration. [HydroGEN PIN portal](https://hydrogen.princeton.edu/pin) · [CDS API setup](https://cds.climate.copernicus.eu/how-to-api)

```bash
python -m LSTM_model.data.conus_observations download
python -m LSTM_model.data.conus_observations select
```

The raw observations live in `inputs/CONUS/observations/`. The analysis searched all windows of at least 60 consecutive months in the full daily archive and maximized the number of ERA5-Land cells with continuous monthly anomalies. January 2020 through December 2024 was the unique maximum with a fixed set of contributing wells. The selection command reproduces that period in `inputs/CONUSmonthly/`. Each monthly well mean uses at least one finite daily measurement on the well's local calendar. Each selected well has 60 monthly means with variation across years for every calendar month.

## 2. Grid the observations and CONUS2 statics

```bash
python -m LSTM_model.data.conus_inputs static-conus2
python -m LSTM_model.data.conus_era5land_grid download-soil
python -m LSTM_model.data.conus_era5land_grid grid
```

These commands reuse the full ParFlow CONUS2 snapshot in `inputs/CONUSmonthly/static/conus2/`. Each ERA5-Land cell takes both slopes and the surface soil parameters from the nearest CONUS2 cell in the Lambert conformal projection. The crosswalk compares x-direction permeability, porosity, and van Genuchten alpha and N for textures. It compares log-permeability across all six TCL bedrock classes for bedrock. Match distances are in `soil_mapping.csv`. [CONUS2 domain catalog](https://hf-hydrodata.readthedocs.io/en/latest/gen_conus2_domain.html)

Wells map to the nearest ERA5-Land cell. Each monthly cell value is the equal-weight mean of its qualifying wells. `inputs/CONUSmonthly/era5land_grid/observations_gridded.csv` retains all **801 observation cells** from **1,005 wells**, for **48,060 cell-months**. Of these, **786 cells** passed the static checks.

## 3. Sample forcing and prepare inference

```bash
python -m LSTM_model.data.conus_era5land_arco --keep-leap-days
python -m LSTM_model.data.conus_inputs prepare
python -m LSTM_model.data.conus_inputs preflight
```

The forcing stage reads official ECMWF ERA5-Land ARCO Zarr directly. It averages 24 hours of 2 m temperature, 2 m dewpoint, and volumetric soil water layer 3, and sums precipitation increments from 01–23 UTC to match the training predictor. Raw sampling covers 1 January 2019 through 31 December 2024. Preparation converts precipitation to millimetres, computes VPD in kPa, retains volumetric soil moisture, and removes leap dates. The model bundle has **365 lookback days and 1,825 daily targets**. [Copernicus ARCO access](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land-timeseries?tab=analysis_ready_data) · [ECMWF accumulation conventions](https://confluence.ecmwf.int/pages/viewpage.action?pageId=462529323)

The prepared inputs are in `inputs/CONUSmonthly/era5land_grid/prepared/`. Two cells with complete statics had gaps in ERA5-Land forcing, leaving **784 prediction cells** and **984 contributing wells**. The static and forcing exclusion CSVs record the removed cells. Preflight checks the feature arrays, the partition of observation and excluded cells, and compatibility with all 100 checkpoints and training scalers.

## 4. Predict

```bash
bash submit_conusmonthly_prediction.sh
```

The current script requests one JURECA-DC node with four GPUs on `dc-gpu-devel` for 10 minutes and runs members 000–099 in order. For a full run with a longer allocation, use `bash submit_conusmonthly_prediction.sh --partition=dc-gpu --time=05:00:00`. Each member writes a daily WTD array and JSON sidecar to `outputs/CONUSmonthly/predictions/member_NNN/`. A resubmission skips members that already have both output files.

Direct submission with `sbatch submit_conusmonthly_prediction.sh` also works from the repository root. Direct submission uses Slurm's default `slurm-<jobid>.out` log. The Bash command uses `outputs/CONUSmonthly/logs/prediction_<jobid>.out` and `.err`.

## 5. Evaluate monthly anomalies

```bash
python -m LSTM_model.data.conusmonthly_evaluation
```

The evaluator checks all 100 predictions against the prepared dates and cell order. It averages daily predictions within each month, then computes separate calendar-month z-scores for the observations and each member using the five-year climatology and population standard deviation. It averages the member anomalies after standardization. Constant calendar months yield undefined anomalies.

The default results are in `outputs/CONUSmonthly/evaluation/run_2020_2024/`. The run writes aligned NetCDF series, daily sample counts per month, per-cell RMSE, MAE, Pearson r, NSE and anomaly-specific KGE, time series, maps, CDFs and ensemble statistics plots. Add `--plot-cells 0 10` to limit individual figures or `--output` to choose a fresh result directory.
