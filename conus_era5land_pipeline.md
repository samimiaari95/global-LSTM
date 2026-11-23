# CONUS groundwater evaluation with ERA5-Land

This workflow applies the Europe-pretrained 100-member LSTM ensemble to USGS groundwater observations on the native ERA5-Land grid. HydroData's ParFlow CONUS2 grid supplies the static predictors. Run all commands from the repository root on JURECA.

## Environment and credentials

```bash
source /p/project1/cslts/miaari1/python_scripts/DailyScriptBox/envs/loadenv.JURECA-DC_JUWELS_2025_GCC-OpenMPI_Std-Py-AI.ini
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
```

When the HydroData, CDS or Zarr clients are missing, install the compatible packages with the activated interpreter. The prediction job uses the base project environment.

```bash
conus_dependencies="$(mktemp -d /tmp/conus-downloads.XXXXXX)"
python -m pip install --target "$conus_dependencies" \
  'numpy<2' 'pandas<3' 'xarray==2024.11.0' 'scipy<1.16' \
  'hf_hydrodata==1.4.17' 'cdsapi==0.7.7' 'zarr==2.18.7' 'aiohttp>=3.10'
export PYTHONPATH="$conus_dependencies${PYTHONPATH:+:$PYTHONPATH}"
```

HydroData prompts for the registered account's PIN or reads `CONUS_HYDRODATA_PIN` for unattended use. Configure CDS with an authorized `$HOME/.cdsapirc` file or point `CDSAPI_RC` at an existing configuration. [HydroGEN PIN portal](https://hydrogen.princeton.edu/pin) · [CDS API setup](https://cds.climate.copernicus.eu/how-to-api)

## 1. Download and select monthly observations

```bash
python -m LSTM_model.data.conus_observations download
python -m LSTM_model.data.conus_observations select --latest-end 2026-11-17
```

The downloader archives daily USGS NWIS water-table depth in metres for the contiguous US and DC. The current archive contains 3,979 wells in 47 state or district responses. Monthly WTD in this run comes from the mean of available daily records in each month, using the site's local calendar dates. Each monthly mean requires at least one recorded day. The source archive lives under `inputs/CONUS/observations/`. [NWIS data and units](https://hf-hydrodata.readthedocs.io/en/latest/gen_usgs_nwis.html)

The selector ranks windows by well count. Each included well must have at least 24 consecutive months of finite WTD and enough variation for a z-score in every calendar month. Ties favor more months, then the earliest start. The default end-date cutoff is six UTC days before selection to allow for ERA5-Land publication lag. The fixed cutoff above reproduces **March 2022–February 2024**, with **1,258 wells**. The outputs under `inputs/CONUS/monthly_anomaly_run/` are `selection.json`, `candidate_windows.csv`, `selected_stations.csv`, and `selected_wtd.csv`. For an independent selection, specify an existing `--archive` and a new `--root`. [Point metadata fields](https://hf-hydrodata.readthedocs.io/en/latest/available_metadata.html)

## 2. Download CONUS2 statics and grid the wells

```bash
python -m LSTM_model.data.conus_inputs static-conus2
python -m LSTM_model.data.conus_era5land_grid download-soil
python -m LSTM_model.data.conus_era5land_grid grid
```

The static downloader saves the full 1 km CONUS2 snapshot under `static/conus2/`. The grid has 3,256 rows and 4,442 columns. Its source fields include both slopes, latitude, longitude, the active-domain mask, and ten soil indicator layers in bottom-to-top order. The soil download adds hydraulic parameters at layer index 9, the top 0.1 m. [HydroData CONUS2 domain catalog](https://hf-hydrodata.readthedocs.io/en/latest/gen_conus2_domain.html)

The gridding step assigns each well from its coordinates to the nearest point on the native 0.1° ERA5-Land grid. It writes `well_era5land_cells.csv` and a separate `observations_gridded.csv` under `inputs/CONUS/monthly_anomaly_run/era5land_grid/`. The latter contains equally weighted monthly mean WTD and the contributing well count for **all 977 cells**, totaling **23,448 cell-month rows**.

Each ERA5-Land cell center selects one nearest CONUS2 cell center in the CONUS2 Lambert conformal projection. Both slopes, the source soil code, and the hydraulic parameters come from that single cell. Prediction uses ERA5-Land coordinates. Indices `(y, x)` refer to ERA5-Land, while `conus2_y` and `conus2_x` identify the sampled static cell.

The soil crosswalk matches CONUS2 texture codes 1–13 to `sample_MapIndicatorToSoilProperties.tcl` using x-direction permeability, porosity, and van Genuchten alpha and N. Distance is the root mean square of differences scaled by the reference-class standard deviation for each parameter. Permeability and alpha use log transforms. Residual saturation is saved for audit because its parameterization differs between the two models. For CONUS2 bedrock codes 19–28, the workflow compares log-permeability across all six TCL bedrock classes. `soil_mapping.csv` records the source cells, parameters, chosen codes, and match distances. [HydroFrame CONUS2 model template](https://github.com/hydroframe/subsettools/blob/main/src/subsettools/template_runscripts/conus2_spinup_solid.yaml)

## 3. Download ERA5-Land forcing and prepare prediction inputs

```bash
python -m LSTM_model.data.conus_era5land_arco --keep-leap-days
python -m LSTM_model.data.conus_inputs prepare
python -m LSTM_model.data.conus_inputs preflight
```

The forcing download reads the official ECMWF ERA5-Land ARCO Zarr archive at `arco.datastores.ecmwf.int` using the CDS API key. Direct archive reads bypass the Copernicus request queue. The downloader averages 24 UTC hours for 2 m temperature, 2 m dewpoint, and volumetric soil water layer 3. It sums hourly precipitation increments from 01 through 23 UTC to reproduce the European workflow's 23 UTC predictor. The 00 UTC increment belongs to the preceding forecast. The raw archive contains 1,096 dates from 2021-03-01 through 2024-02-29, including the 365-day lead-in and leap day. Its companion JSON records the date range and completeness by variable. [Copernicus Analysis Ready Data](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land-timeseries?tab=analysis_ready_data) · [ECMWF accumulation conventions](https://confluence.ecmwf.int/pages/viewpage.action?pageId=462529323)

Preparation selects the model's no-leap calendar, converts precipitation to millimetres, computes VPD in kPa from daily temperature and dewpoint, and preserves volumetric soil moisture. Dynamic array order is `(feature, context_day, cell)` for precipitation, VPD and soil moisture. Static array order is `(feature, cell)` for slope x, slope y, mapped soil indicator, longitude and latitude. Prediction reads cell IDs from `prepared/selected_pixels.npy`. Use a fresh run directory to restage inputs.

The nearest CONUS2 cells had missing static values for **19 coastal ERA5-Land cells**. Another **3 cells** had missing ERA5-Land forcing first detected in March 2021, during the lookback. The two exclusion CSVs record their indices and coordinates, and the observation archive retains their WTD. The prepared bundle contains **955 cells**, **1,229 wells**, and **730 daily prediction targets**. CPU preflight checks the cell partition, feature arrays, checkpoints and scaler compatibility for all 100 members.

The prepared soil array contains 11 whole-number TCL codes stored as `float32`, ranging from 1 to 14. The 13 bedrock cells use `BGR1`, code 14. All 11 codes occur in the supplied European soil file. Code 5 covers 38.64% of the selected cells and 0.31% of the full European grid. Members use their original training means and standard deviations. Longitude, soils and climate differ across the regions, so the scores describe an out-of-region transfer.

## 4. Generate daily predictions when authorized

```bash
mkdir -p outputs/CONUS/era5land_grid/logs
sbatch job_conus_prediction.sh
```

The launcher defaults to the native ERA5-Land `prepared/` bundle. It uses JURECA's `dc-gpu` partition, account `jicg41`, four GPUs, 128 CPUs and a five-hour limit per task. Slurm runs one pretrained member per array task, with up to ten simultaneous tasks. Each member writes a daily WTD array and JSON sidecar under `outputs/CONUS/era5land_grid/predictions/member_NNN/`. The launcher accepts explicit input and prediction directories as its first and second arguments.

## 5. Evaluate and plot

```bash
python -m LSTM_model.data.conus_evaluation \
  --output outputs/CONUS/era5land_grid/evaluation/run_001
```

Evaluation checks all 100 member arrays against the prepared directory and averages daily predictions by month. It averages observed wells sharing each retained cell. It standardizes observations and each member separately with calendar-month climatologies and population standard deviations over the common period, then averages the standardized members. With two values per calendar month, a varying month produces a pair of −1 and +1 anomalies. Constant months yield undefined anomalies. This limits interpretation of amplitude. Well sampling, pumping, aquifer depth, and point-to-grid representativeness also affect the comparison.

Outputs include aligned NetCDF series, per-cell RMSE, MAE, Pearson r, NSE and anomaly-specific KGE, maps, time series, scatter plots, score CDFs, and spread-versus-skill panels. Use `--plot-cells 0 10` to limit the individual figures. Evaluation needs a new output directory and temporary disk space for the member anomalies.
