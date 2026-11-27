#!/bin/bash
set -eo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/../.."
source /p/project1/cslts/miaari1/python_scripts/DailyScriptBox/envs/loadenv.JURECA-DC_JUWELS_2025_GCC-OpenMPI_Std-Py-AI.ini
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -u -m LSTM_model.coordinate_sensitivity.prepare prepare "$@"
