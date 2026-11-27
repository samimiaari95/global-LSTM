# ERA5-Land LSTM ensemble statistics maps

Predict daily water table depth (WTD) on the masked European EUR11 grid using
the pretrained 100-member LSTM ensemble. Predictions cover 2017 through 2020
on the model's 365-day calendar. Each target uses the preceding 365 days of
features. Preprocessing reads forcing from 2016 through 2020 to supply the
lookback for the first target.

## Inputs

### Mask used for selected pixels

- File: `inputs/EU_74_-48_69_20/mapping_0stdroll6months.npy`
- Purpose: binary mask selecting valid pixels for prediction.
- Rule: only predict for pixels where the array value is exactly `1`.
- All other pixels must remain excluded from the statistics-map predictions.

### ERA5-Land regridded inputs

Use the regridded daily ERA5-Land files already prepared for the EU domain, typically under a structure similar to:

- `inputs/EU_74_-48_69_20/raw/ERA5-Land/regridded_to_EUR11/`
- or the equivalent regridded directory already used by the project scripts

The prediction pipeline must:

- load the relevant daily ERA5-Land variables,
- align to the EU grid,
- keep only the selected pixels from the mask,
- convert them to the exact format expected by the pre-trained ensemble model,
- generate daily WTD predictions for each selected pixel.

Member checkpoints and fitted scalers live under
`LSTM_model/model/ensemble_400px/` as configured in [config.py](../LSTM_model/model/config.py).
The pipeline stages are implemented in
[`LSTM_model/statistics_maps/`](../LSTM_model/statistics_maps/).

## Three-stage workflow

Run these commands from the repository root on JURECA. The helpers save Slurm job IDs inside the run
directory and use them internally for dependencies, so you only pass the run
name between commands:

```bash
bash scripts/statistics_maps/submit_statisticsmap_preprocess.sh stats_2017_2020
```

If no run name is supplied, the script creates a unique timestamped name and
refuses to reuse an existing run directory. Submit the 100 independent
prediction jobs with the same run name:

```bash
bash scripts/statistics_maps/submit_statisticsmap_prediction.sh stats_2017_2020
```

The helper loops over member indices 0-99 and runs `sbatch` once per member.
Each submission sets `MEMBER_INDEX` for that job, requests four GPUs, and
writes a member-specific log. Slurm schedules the 100 jobs according to account limits and GPU availability. If all run at once, they request up to 400 GPUs.

If preprocessing was submitted with the preprocessing helper, each prediction
job gets an `afterok` dependency on preprocessing using its saved job ID. If
preprocessing was performed outside the helper, the prediction helper verifies
the prepared input files exist and submits without a preprocessing dependency.
The helper records each member index and its job ID in
`submission_state/prediction_jobs.tsv`.

Finally, submit postprocessing with just the run name:

```bash
bash scripts/statistics_maps/submit_statisticsmap_postprocess.sh stats_2017_2020
```

The helper validates the recorded member/job mapping. When prediction outputs
are already complete, it submits postprocessing without a Slurm dependency
(avoiding dependency errors when Slurm has purged old completed jobs). If
some members are incomplete but their jobs remain queued or running, it adds
`afterok` dependencies on those jobs.
Run directories and logs are stored under
`outputs/EU_74_-48_69_20/statisticsEUmap/stats_2017_2020/`.


## Prepared features

### Input structure must match `utils.transferpx_inputfeatures`

The new pipeline must follow the same LSTM input construction used in `LSTM_model/utils/utils.py::transferpx_inputfeatures`.

The exact required pattern is:

1. For each ERA5-Land feature file:
   - load the raw data array,
   - reshape to `(timeseries, cells)` if it is 3D and the project stores one spatial dimension as a flat cell vector,
   - keep only the selected valid pixels for the current batch,
   - move the time axis so the variable becomes `(cells, timeseries)`.

2. Standardize each feature using the same running statistics logic as the project:
   - mean and std are computed from the valid data region,
   - values are transformed as:
     `data = (data - mean) / std`

3. Build the lookback window exactly as in the project code:
   - for each time index `i` in the allowed window range,
   - take the previous `LOOKBACK` days for each cell,
   - create a windowed feature tensor shaped like `(cells, LOOKBACK)`,
   - then reshape to the form expected by the LSTM ensemble.

4. Stack the feature tensors across all predictor variables in the same order as `FEATURES_FILES`.

In the project implementation, this is equivalent to:

```python
for inputvar in FEATURES_FILES:
    raw_data = np.load(path_to_inputvar)
    raw_data = raw_data.reshape(raw_data.shape[0], NB_CELLS) if len(raw_data.shape) > 2 else raw_data
    data = raw_data[start:end, :]
    data = np.moveaxis(data, 0, -1)  # (cells, timeseries)
    data = (data - mean) / std
    lookback_arrays = [data[:, i-LOOKBACK:i] for i in range(LOOKBACK, end-start)]
    lookback_arrays = np.array(lookback_arrays)
    f1 = lookback_arrays.reshape(-1, LOOKBACK)
    f1 = np.array([f1])
    all_inputs = np.concatenate((all_inputs, f1), axis=0) if len(all_inputs) > 0 else f1

all_inputs = np.moveaxis(all_inputs, 0, -1)
```

The statistical-map prediction script must follow the same logic, except that the target cells are limited to the pixels where `mapping_0stdroll6months.npy == 1` and the processing is batched to keep memory usage low.

This is the key requirement for matching the trained ensemble input structure.
The prepared feature file stores the windows directly in
`(prediction_days * selected_pixels, LOOKBACK, features)` order: prediction
date first, selected pixel second, then lookback day and feature. Feature
standardization remains member-specific and is applied by the prediction job.
For the current 43,226 selected pixels and 1,460 prediction days, this
float32 `.npy` file is approximately 686 GiB; preprocessing checks reported
free filesystem space before creating it. The preprocessing job requests three
hours to allow for writing this expanded window array; actual wall time depends
on shared-filesystem throughput and any project quota.


## Run files

| Path under `outputs/EU_74_-48_69_20/statisticsEUmap/<run_name>/` | Contents |
| --- | --- |
| `feature_inputs/` | Precomputed feature windows |
| `mask/` | Selected pixels and grid metadata |
| `predictions/` | Daily WTD arrays by member |
| `metadata/` | Input dates, preprocessing metadata, and per-member completion manifests |
| `maps/` | Reconstructed EUR11 statistics with NaN outside the mask |
| `submission_state/` and `logs/` | Slurm job IDs and logs |
