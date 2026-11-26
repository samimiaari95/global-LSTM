# European ERA5-Land inputs for the LSTM ensemble

The workflow samples fresh forcing from the ECMWF ERA5-Land ARCO geo-chunked Zarr stores. It uses the cells with complete monthly `wtd_a` observations from January 1996 through December 2012 in the [native-grid observation NetCDF](../inputs/EU_74_-48_69_20/localobservations/wtd_anomalies_era5land_0p1deg.nc). Each selected cell retains its native ERA5-Land row, column, latitude, and longitude. [ECMWF ARCO product guide](https://confluence.ecmwf.int/pages/viewpage.action?pageId=536218894)

## Run on JURECA

From the repository root, activate the project environment. Load the NetCDF Python module and make the download libraries available to that interpreter.

```bash
source /p/project1/cslts/miaari1/python_scripts/DailyScriptBox/envs/loadenv.JURECA-DC_JUWELS_2025_GCC-OpenMPI_Std-Py-AI.ini
module load netcdf4-python
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

era5land_deps="$(mktemp -d /tmp/eu-arco-deps.XXXXXX)"
python -m pip install --target "$era5land_deps" \
  'numpy<2' 'cdsapi==0.7.7' 'zarr==2.18.7' 'aiohttp>=3.10,<4'
export PYTHONPATH="$era5land_deps${PYTHONPATH:+:$PYTHONPATH}"
```

Configure a CDS personal access token in `~/.cdsapirc` or point `CDSAPI_RC` to the credentials file. The downloader uses that token for authenticated access to the ECMWF stores. [CDS API setup](https://cds.climate.copernicus.eu/how-to-api)

```bash
python -u -m LSTM_model.europe.prepare_era5land_ensemble_inputs all --workers 12
```

The [Python workflow](../LSTM_model/europe/prepare_era5land_ensemble_inputs.py) also takes the individual stages `select`, `download`, `prepare`, and `verify`. The default output root is `inputs/EU_74_-48_69_20/era5land_arco_1996_2012/`. An interrupted download can resume with `download` in the same root, using tiles fetched for the same dates and cells. Use a new `--root` to download again from scratch. `--observations` and `--static-dir` override the inputs for `select`.

## Predictors and dates

- The hourly pull covers 1 January 1995 through 31 December 2012 UTC. The first year provides the lookback for the 1 January 1996 prediction. The raw daily point archive keeps leap days. Preparation drops 29 February to match the ensemble’s 365-day calendar, leaving 6,570 forcing dates and 6,205 prediction dates.
- Daily precipitation sums the **01:00 through 23:00 UTC** hourly increments to match the ensemble’s 23 UTC accumulation convention and converts metres to millimetres. Soil moisture is the 24-hour mean for ERA5-Land soil layer 3. VPD in kPa is the difference in saturation vapour pressure computed from the daily means of 2 m air temperature and dew point. [ERA5-Land accumulation conventions](https://confluence.ecmwf.int/pages/viewpage.action?pageId=462529323)
- Each ERA5-Land cell takes `slopex`, `slopey`, and `soilind` from one nearest static donor by great-circle distance. The donor coordinates come from `lon2D.npy` and `lat2D.npy`. The code reads the first plane of each time-repeated static raster. The longitude and latitude features are the selected ERA5-Land cell centers.
- A cell enters the prepared bundle only when all three daily model features are finite throughout the model calendar. `excluded_forcing_pixels.csv` lists any incomplete cells by ERA5-Land indices and gives the first missing date for each affected feature. No spatial reassignment or imputation is applied.

## Model input layout

Each member sees a logical float32 array of shape **`(6,205 × retained_pixels, 365, 8)`**. The feature order is precipitation, VPD, soil moisture layer 3, x slope, y slope, soil indicator, ERA5-Land longitude, ERA5-Land latitude. For a target at index `t`, the window uses input dates `t` through `t + 364` on the context axis, ending one model day before the target. Samples are ordered by target date, then by ERA5-Land row and column. Sample `i` maps to `divmod(i, retained_pixels)`.

The prepared files work directly with `iter_member_input_batches` in [the existing ensemble preprocessor](../LSTM_model/model/ensemble_inputs.py). The iterator builds arrays of shape `(batch_samples, 365, 8)` and applies each member’s training means and standard deviations. This avoids writing a separate full windowed tensor for each of the 100 members. Pass the bundle to `python -m LSTM_model.model.predict_era5land_observations` via `--prepared-directory` when launching distributed inference.

| Path under the output root | Contents |
| --- | --- |
| `observation_pixels.csv` | All qualifying observation cells and their static values, donor indices, and donor distances |
| `arco/daily_points.npz` | Fresh daily `tp`, `t2m`, `d2m`, and `swvl3` arrays in source units, in date × cell order |
| `arco/tiles_*/` | Resumable downloads reduced to daily values for the requested cells |
| `prepared/dynamic_features.npy` | `(3, 6570, retained_pixels)`, daily model features in physical units |
| `prepared/static_features.npy` | `(5, retained_pixels)`, in model feature order |
| `prepared/selected_pixels.npy` and `selected_pixels.csv` | Retained ERA5-Land row and column indices, with coordinates and static donor information in the CSV |
| `prepared/context_dates.npy` and `target_dates.npy` | Date axes in `datetime64[D]` |
| `prepared/excluded_forcing_pixels.csv` | Cells omitted from the inference bundle and first missing dates |
| `prepared/preprocessing.json` | Source paths, feature order and units, logical tensor shape, and counts |

The `verify` stage checks date axes, finite feature arrays, all 100 feature-scaler files, and the exact normalized values at the first, next-day, and last sample positions.

## Verified European run

The observation mask contains 1,873 cells with 204 finite monthly anomalies each. The download sampled all 1,873 locations. All three daily model features have complete coverage at 1,861 cells. The 12 omitted cells have missing values in all three features beginning on 1 January 1995.

The prepared dynamic array is `(3, 6570, 1861)` and the static array is `(5, 1861)`. The ensemble reads **`(11547505, 365, 8)`** per member through the batch iterator. All 100 training feature scalers passed validation.
