# Monthly verification of ERA5-Land groundwater predictions

Run from `global_TL/global-LSTM` on JURECA. The script reads the completed predictions and runs on one CPU process. It uses NumPy, pandas, SciPy, xarray, Matplotlib, pyshp, Shapely and netCDF4. It calls the project KGE utility, whose configuration imports PyTorch. Computation and rendering use CPU arrays and the Agg backend.

```bash
source /p/project1/cslts/miaari1/python_scripts/DailyScriptBox/envs/loadenv.JURECA-DC_JUWELS_2025_GCC-OpenMPI_Std-Py-AI.ini
module load netcdf4-python
export CUDA_VISIBLE_DEVICES="" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m LSTM_model.europe.postprocess_era5land_observations
```

Results go under `outputs/EU_74_-48_69_20/era5land_arco_1996_2012/observation_anomalies/run_<timestamp>/`. Use `--output PATH` to choose a new directory. Relative paths start at the working directory. The script publishes the results directory after the figures and tables finish. Allow roughly 1 GiB of free disk space and 1–2 GiB of RAM. Daily predictions total about 4.3 GiB and are memory-mapped one member at a time.

## Inputs and alignment

- Member IDs 0–99 each supply a `daily_wtd.npy` array and matching JSON in `outputs/EU_74_-48_69_20/era5land_arco_1996_2012/predictions/member_NNN/`. The sidecars identify the native ERA5-Land grid, prediction dates, and ordered cell indices. A missing member, incompatible metadata, or nonfinite daily output stops the run.
- `inputs/EU_74_-48_69_20/era5land_arco_1996_2012/prepared/` supplies the dates, ordered row and column indices, and preprocessing metadata. Override these directories with `--prepared-directory` and `--predictions-directory`.
- The observation source is the native-grid NetCDF named in `preprocessing.json`. The script uses the same latitude and longitude indices as the predictions, aligns observations by year and month, and uses the exact grid coordinates for maps. Each evaluated cell has one selected well and 204 monthly observations. The 12 cells excluded during forcing preparation are outside the evaluation.

## Calculation

1. Average the destandardized daily predictions within each year-month for each member and cell. Every available model day has equal weight. The calendar contains all 6,205 dates in 1996–2012 except February 29. Each February therefore uses 28 predictions.
2. Following the anomaly calculation in `postprocess.py`, calculate a separate climatology for each member, cell, and calendar month. Subtract the 17-year mean for that month and divide by its population standard deviation (ddof 0). Average these standardized anomalies across all 100 members with equal weights. Member normalization occurs before ensemble averaging, which affects the weight given to members with different amplitudes. Preserve exact zero anomalies. If a member has constant values for a calendar month, exclude that cell’s ensemble anomaly for that calendar month across all years. The raw monthly ensemble mean remains available.
3. Use the supplied dimensionless `wtd_a` anomalies directly with their source climatologies.
4. Score the ensemble mean at each cell over months with finite observed and predicted anomalies. Scores are RMSE, MAE, Pearson r, Nash–Sutcliffe efficiency (NSE) and the project's anomaly-specific Kling–Gupta efficiency (KGE). The script calls `utilities.calculate_kge` from `LSTM_model/utils/utils.py`. Its beta term is fixed at 1, so KGE measures correlation and the ratio of predicted to observed temporal standard deviation. Pearson r and KGE require both series to vary.
5. The predictive R² displayed in the scatter plots is one minus the sum of squared prediction errors divided by the sum of squared observed departures from the observed mean. It can be negative. At one cell this equals that cell’s NSE. The global calculation pools finite cell-months and uses their pooled observed mean. Cells with more valid months receive more weight. R² is undefined with fewer than two pairs or constant observations.
6. Calculate ensemble statistics on those same paired months. For each month compute the population variance, interquartile range (75th minus 25th percentile), population SD and mean absolute deviation of members about their ensemble mean. Average each statistic across time. Pairwise r is the average of the 4,950 distinct member-to-member temporal correlations. It is undefined if any participating member has a constant paired series. RMS_spread, also reported, is the square root of the time-average across-member sample variance (ddof 1).
7. Climatologies use the evaluation period itself. Scores describe retrospective anomaly skill. Spatial dependence, uneven station coverage and temporal autocorrelation affect the interpretation of the scatter plots and summaries.

## Outputs

| File or directory | Contents |
| --- | --- |
| `monthly_series.nc` | Monthly ensemble mean in native WTD units, standardized ensemble mean, observed anomaly, across-member sample SD, dates, contributing day counts, cell indices and coordinates. |
| `cell_metrics.csv`, `metric_summary.csv` | Cell scores, ensemble statistics, coverage and coordinates, plus equal-cell descriptive summaries of the five metrics. Undefined values are blank. |
| `timeseries/cell_NNNN.png` | One file per selected cell. Gray transparent lines show individual standardized members, blue shows observations and orange shows the ensemble mean. The title gives all five scores, the y-axis reads WTD anomalies, and a locator map highlights the cell. Gaps mark undefined values. |
| `cell_r2/cell_NNNN.png` | One observed-versus-ensemble-mean scatter per cell, with predictive R², paired-month count and a 1:1 reference for finite pairs. The axes identify observed and ensemble-mean WTD anomalies, and a locator map highlights the cell. A cell with no pairs still receives a labeled figure. Cell IDs in both folders match the CSV index. |
| `metric_maps/<metric>.png` | Separate map for each of the five scores. Both panels use the same colors and limits. The left panel covers the full study area. The right spans 6.5°W–19.5°E and 43.85°N–56.25°N, including Czechia and keeping Spain outside the extent. The southern edge also crops southern France. Pearson r, NSE and KGE use −1 to 1. Error scales saturate at the full-area 98th percentile. Exact scores are in the CSV. |
| `statistics_maps.png`, `metric_cdfs.png` | Maps of paired coverage, spread and temporal SD, and empirical cumulative distributions of cell scores. |
| `statistics_vs_metrics.png` | One point per scored cell in each panel. Rows give the performance metric on the vertical axis. Columns give ensemble variance, IQR, pairwise r, mean ensemble SD and MAD on the horizontal axis. The NSE and KGE rows omit points below −1 and −1.41 respectively. CSVs and other plots use the full scores. |
| `paired_anomalies.png` | Global observed-versus-ensemble-mean density, with logarithmically colored cell-month counts, predictive R² and a 1:1 reference. |
| `example_series.png` | Three example series selected by cell RMSE rank, each with a locator map. |
| `well_locations_and_grid_cells.png` | The selected observation wells for evaluated cells and their assigned native ERA5-Land grid centers. |
| `rank_histogram.png` | One pooled histogram over all cell-months with finite observations and all 100 member anomalies. It has 101 ranks, from 0 to 100, and a uniform-count reference. Exact ties use uniform random ranks among the possible tied positions with a fixed seed. |

The default run generates both individual PNGs for **every** selected cell. To inspect selected examples more quickly, append e.g. `--plot-cells 0 174 800`. This limits only the two per-cell figure folders. All summary figures, scores and monthly outputs still use the full ensemble and all cells.

## Verified native-grid run

The [27 November 2026 run](../outputs/EU_74_-48_69_20/era5land_arco_1996_2012/observation_anomalies/run_20261127_native/) evaluated 100 ensemble members at 1,861 cells. All cells contributed 204 months, for 379,644 paired cell-months. Pooled predictive R² was 0.097 and median cell KGE was 0.438. An independent reconstruction from the 100 daily files reproduced the monthly series, RMSE and KGE at cell 800. The full run, including both plots for every cell, took about 13 minutes on one JURECA CPU core.
