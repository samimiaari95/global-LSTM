#!/bin/bash -x
# author: Sami Miaari
# e-mail: s.miaari@fz-juelich.de

#SBATCH --job-name=statsmap_postprocess
#SBATCH --account=jibg36
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=batch
#SBATCH --time=18:00:00
#SBATCH --mail-user=s.miaari@fz-juelich.de
#SBATCH --mail-type=ALL

set -eo pipefail

if [[ -n ${SLURM_JOB_ID:-} ]]; then
    REPOSITORY="$PWD"
else
    REPOSITORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
fi
if [[ ! -f "$REPOSITORY/LSTM_model/model/config.py" ]]; then
    echo "Submit from the repository root or pass sbatch --chdir=/path/to/global-LSTM." >&2
    exit 1
fi
cd "$REPOSITORY"
source /p/project1/cslts/miaari1/python_scripts/DailyScriptBox/envs/loadenv.JURECA-DCadapted_JUWELS_2025_GCC-OpenMPI_Std-Py-AI.ini
set -u
module load netcdf4-python

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

STATISTICS_ROOT="${REPOSITORY}/outputs/EU_74_-48_69_20/statisticsEUmap"
if [[ "$#" -gt 1 ]]; then
    echo "Usage: sbatch $0 [run_name]" >&2
    exit 2
fi
if [[ "$#" -eq 1 ]]; then
    if [[ -n "${RUN_NAME:-}" ]]; then
        if [[ "$RUN_NAME" != "$1" ]]; then
            echo "RUN_NAME and the supplied run_name argument do not match" >&2
            exit 2
        fi
    fi
    RUN_NAME="$1"
fi
if [[ -z "${RUN_NAME:-}" ]]; then
    echo "RUN_NAME is required; use submit_statisticsmap_postprocess.sh or pass a run name:" >&2
    echo "  sbatch $0 <run_name>" >&2
    exit 2
fi
if [[ ! "$RUN_NAME" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "Invalid RUN_NAME: use only letters, numbers, '.', '_' or '-'" >&2
    exit 2
fi

RUN_DIR="${STATISTICS_ROOT}/${RUN_NAME}"
if [[ ! -f "${RUN_DIR}/metadata/preprocessing.json" ]]; then
    echo "Preprocessing metadata is missing for run ${RUN_NAME}: ${RUN_DIR}" >&2
    exit 1
fi

echo "Starting CPU output processing for ${RUN_NAME}"
srun python -u -m LSTM_model.statistics_maps.process_era5land_statisticsmap_outputs \
  --output-root "$STATISTICS_ROOT" \
  --run-name "$RUN_NAME"
