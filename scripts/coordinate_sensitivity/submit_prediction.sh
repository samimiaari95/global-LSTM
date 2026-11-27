#!/bin/bash
# Print the command by default. Submission requires --submit.
set -eo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/../.."
mode="${1:---dry-run}"
run="${2:-outputs/coordinate_sensitivity}"
if [[ "$#" -gt 2 || ( "$mode" != --dry-run && "$mode" != --submit ) ]]; then
    printf 'Usage: %s [--dry-run|--submit] [run_directory]\n' "$0" >&2
    exit 2
fi
run="$(realpath -- "$run")"
source /p/project1/cslts/miaari1/python_scripts/DailyScriptBox/envs/loadenv.JURECA-DC_JUWELS_2025_GCC-OpenMPI_Std-Py-AI.ini
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m LSTM_model.coordinate_sensitivity.prepare verify --run "$run" --check-baseline
mkdir -p "$run/logs"
command=(sbatch --clusters=jureca --parsable --chdir="$PWD" --array=1-3%1
         --output="$run/logs/prediction_%A_%a.out" --error="$run/logs/prediction_%A_%a.err"
         "$PWD/scripts/coordinate_sensitivity/job_prediction.sh" "$run")
printf '%q ' "${command[@]}"
printf '\n'
if [[ "$mode" == --submit ]]; then
    "${command[@]}"
fi
