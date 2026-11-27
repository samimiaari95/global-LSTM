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
python -m LSTM_model.conusmonthly.conus_observations download
python -m LSTM_model.conusmonthly.conus_observations select
```

The raw observations live in `inputs/CONUS/observations/`. The analysis searched all windows of at least 60 consecutive months in the full daily archive and maximized the number of ERA5-Land cells with continuous monthly anomalies. January 2020 through December 2024 was the unique maximum with a fixed set of contributing wells. The selection command reproduces that period in `inputs/CONUSmonthly/`. Each monthly well mean uses at least one finite daily measurement on the well's local calendar. Each selected well has 60 monthly means with variation across years for every calendar month.

## 2. Grid the observations and CONUS2 statics

```bash
python -m LSTM_model.conusmonthly.conus_inputs static-conus2
python -m LSTM_model.conusmonthly.conus_era5land_grid download-soil
python -m LSTM_model.conusmonthly.conus_era5land_grid grid
```

These commands reuse the full ParFlow CONUS2 snapshot in `inputs/CONUSmonthly/static/conus2/`. Each ERA5-Land cell takes both slopes and the surface soil parameters from the nearest CONUS2 cell in the Lambert conformal projection. The crosswalk compares x-direction permeability, porosity, and van Genuchten alpha and N for textures. It compares log-permeability across all six TCL bedrock classes for bedrock. Match distances are in `soil_mapping.csv`. [CONUS2 domain catalog](https://hf-hydrodata.readthedocs.io/en/latest/gen_conus2_domain.html)

Wells keep their existing ERA5-Land cell assignments. Within each cell, the eligible well with the shortest Haversine distance to the cell centre supplies all 60 monthly values. This follows the [European ERA5-Land mapper](../LSTM_model/europe/map_era5land_observations.py) for resolving multiple wells per cell. CONUSmonthly requires a complete series and nonzero variance for every calendar-month climatology. Exact distance ties use the lowest site ID in lexicographic order.

`inputs/CONUSmonthly/era5land_grid/observations_gridded.csv` contains **48,060 cell-months** for one well in each of **801 observation cells**, selected from **1,005 candidates**. `selected_wells_by_cell.csv` records the site ID, centre distance and candidate count per cell. To regenerate these two observation products from the archived assignments and well series, run:

```bash
python -m LSTM_model.conusmonthly.conus_era5land_grid observations
```

## 3. Sample forcing and prepare inference

```bash
python -m LSTM_model.conusmonthly.conus_era5land_arco --keep-leap-days
python -m LSTM_model.conusmonthly.conus_inputs prepare
python -m LSTM_model.conusmonthly.conus_inputs preflight
```

The forcing stage reads official ECMWF ERA5-Land ARCO Zarr directly. It averages 24 hours of 2 m temperature, 2 m dewpoint, and volumetric soil water layer 3, and sums precipitation increments from 01–23 UTC to match the training predictor. Raw sampling covers 1 January 2019 through 31 December 2024. Preparation converts precipitation to millimetres, computes VPD in kPa, retains volumetric soil moisture, and removes leap dates. The model bundle has **365 lookback days and 1,825 daily targets**. [Copernicus ARCO access](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land-timeseries?tab=analysis_ready_data) · [ECMWF accumulation conventions](https://confluence.ecmwf.int/pages/viewpage.action?pageId=462529323)

The prepared inputs are in `inputs/CONUSmonthly/era5land_grid/prepared/`. Of the 801 observation cells, 786 passed the static checks and 784 also passed the forcing checks. These 784 cells contain 984 candidate wells. The nearest-well evaluation selects 784 of them. The static and forcing exclusion CSVs record the removed cells. Preflight checks the feature arrays, the partition of observation and excluded cells, and compatibility with all 100 checkpoints and training scalers.

## 4. Predict

```bash
bash scripts/conusmonthly/submit_conusmonthly_prediction.sh
```

The current script requests one JURECA-DC node with four GPUs on `dc-gpu-devel` for 45 minutes and runs members 000–099 in order. For a full run with a longer allocation, use `bash scripts/conusmonthly/submit_conusmonthly_prediction.sh --partition=dc-gpu --time=05:00:00`. Each member writes a daily WTD array and JSON sidecar to `outputs/CONUSmonthly/predictions/member_NNN/`. A resubmission skips members that already have both output files.

Direct submission with `sbatch scripts/conusmonthly/submit_conusmonthly_prediction.sh` also works from the repository root. Direct submission uses Slurm's default `slurm-<jobid>.out` log. The Bash command uses `outputs/CONUSmonthly/logs/prediction_<jobid>.out` and `.err`.

## 5. Evaluate monthly anomalies

```bash
python -m LSTM_model.conusmonthly.conusmonthly_evaluation
```

The evaluator verifies that nearest-well observations cover exactly the prepared grid, then checks all 100 predictions against the prepared dates and cell order. The mapping uses the same 784 cells and 1,825 daily targets, so this evaluation reuses the existing predictions. It averages daily predictions within each month, then computes separate calendar-month z-scores for the selected wells and each member using the five-year climatology and population standard deviation. It averages the member anomalies after standardization. Constant calendar months yield undefined anomalies.

The [default output](../outputs/CONUSmonthly/evaluation/run_2020_2024_nearest_well/) contains aligned NetCDF series, the selected site ID and its centre distance for each cell, daily sample counts per month, per-cell RMSE, MAE, Pearson r, NSE and anomaly-specific KGE, metric maps, CDFs and ensemble statistics plots. The [averaged-well evaluation](../outputs/CONUSmonthly/evaluation/run_2020_2024_maps_and_ranks/) is available for comparison.

- `rank_histogram.png` pools observations across all evaluated cells and months. A cell-month contributes when the observed anomaly and all 100 member anomalies are finite. Ranks 0–100 count the members below the observation, with ties broken reproducibly at random. A dashed line shows the count expected under a uniform rank distribution.
- `timeseries/` and `cell_r2/` label the plotted values as WTD anomalies and show each cell in a small CONUS map. The example time series also show cell locations.
- `well_locations_and_grid_cells.png` shows all 1,005 point observation wells in one style on the left. The right panel shows the 801 assigned ERA5-Land cell centres, split into 784 valid cells and 17 cells that failed static or forcing checks. For a standalone prepared bundle, place `well_era5land_cells.csv` in its parent directory to show all wells.
- `statistics_vs_metrics.png` shows RMSE, Pearson r, NSE and KGE against time-averaged ensemble variance (EV with an overbar), time-averaged interquartile range (IQR with an overbar), pairwise member correlation and MAD.
- Each metric map uses the metric name as its title.

Add `--plot-cells 0 10` to limit individual figures. The pooled rank histogram and the coverage figure still use the full applicable dataset. Use `--output` to choose a fresh result directory.
