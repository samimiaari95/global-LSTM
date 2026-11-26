# global-LSTM

This repository trains a spatiotemporal LSTM for water table depth and runs
ensemble prediction and evaluation over CONUS and Europe.

## Repository layout

```text
LSTM_model/
  conusmonthly/       Monthly CONUS workflow and common CONUS input preparation
  conusdaily/         Daily CONUS evaluation and instantaneous-forcing experiment
  europe/             European inputs, training, validation, and observation evaluation
  statistics_maps/    European gridded ensemble prediction and postprocessing
  model/              Model configuration and shared ensemble inference
    ensemble_400px/   Local checkpoints and fitted scalers for members 0–99
  utils/              Shared data, statistics, and plotting helpers
scripts/              Shell and Slurm scripts organized by project
documentations/       Markdown pipeline and workflow guides
inputs/               Local input data
outputs/              Predictions, figures, and other generated results
main.py               European training and validation entry point
```

## Setup

```bash
git clone https://github.com/samimiaari95/global-LSTM.git
cd global-LSTM
git clone --recurse-submodules https://github.com/HPSCTerrSys/SLOTH.git
```

On JURECA or JUWELS, activate the project environment before running Python.

```bash
source /p/project1/cslts/miaari1/python_scripts/DailyScriptBox/envs/loadenv.JURECA-DC_JUWELS_2025_GCC-OpenMPI_Std-Py-AI.ini
module load h5py scikit-learn Seaborn netcdf4-python
```

For local setup, install the packages in `requirements.txt` in your Python environment. Run
the workflow commands from the repository root.

The pretrained weights and scalers are local assets. On a fresh checkout,
copy each member's `400px_member_<id>_<model_name>.pt` and
`meanstd_400px_member_<id>_<model_name>.pkl` to
`LSTM_model/model/ensemble_400px/400px_member_<id>/`. Keep the member
number and model-name suffix from the existing files. The previous layout stored
these assets beside European member predictions under
`outputs/EU_74_-48_69_20/validation_ERA5/ensemble_400px/`.

## Workflows

| Project | Guide |
| --- | --- |
| CONUSmonthly | [Preparation, prediction, and evaluation](documentations/conusmonthly_pipeline.md) |
| CONUSdaily | [Daily evaluation and prediction](documentations/conusdaily_pipeline.md) |
| Europe | [Native ERA5-Land inputs](documentations/era5land_ensemble_inputs.md) and [EUR11 observation postprocessing](documentations/era5land_observation_postprocessing.md) |
| Statistics maps | [Preprocessing, distributed prediction, and map output](documentations/statisticsmapinstructions.md) |

For the European EUR11 workflow, `python main.py --ERA5inputs` prepares
observation-cell predictors, and
`bash scripts/europe/jurecaGPU_predict_era5land.sh` submits ensemble
inference. `python main.py --help` lists the other European entry points,
including training and validation.
