#!/bin/bash -x
# author: Sami Miaari
# e-mail: s.miaari@fz-juelich.de

#SBATCH --job-name=statsmap_preprocess
#SBATCH --account=jibg36
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=batch
#SBATCH --time=03:00:00
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
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"

STATISTICS_ROOT="${REPOSITORY}/outputs/EU_74_-48_69_20/statisticsEUmap"
if [[ -z "${RUN_NAME:-}" ]]; then
    RUN_NAME="run_preprocess_${SLURM_JOB_ID:-manual}_$(date +%Y%m%d_%H%M%S)"
    RUN_DIR="${STATISTICS_ROOT}/${RUN_NAME}"
    mkdir -p "$STATISTICS_ROOT"
    mkdir "${RUN_DIR}"
    mkdir "${RUN_DIR}/logs"
    echo "RUN_NAME was not supplied; created standalone run ${RUN_NAME}"
else
    if [[ ! "$RUN_NAME" =~ ^[A-Za-z0-9._-]+$ ]]; then
        echo "Invalid RUN_NAME: use only letters, numbers, '.', '_' or '-'" >&2
        exit 2
    fi
    RUN_DIR="${STATISTICS_ROOT}/${RUN_NAME}"
    if [[ ! -d "${RUN_DIR}/logs" ]]; then
        echo "Run directory must be created by submit_statisticsmap_preprocess.sh: ${RUN_DIR}" >&2
        exit 1
    fi
fi

echo "Starting CPU preprocessing for ${RUN_NAME}"
srun python -u -m LSTM_model.statistics_maps.prepare_era5land_statisticsmap \
  --mask-path "${REPOSITORY}/inputs/EU_74_-48_69_20/mapping_0stdroll6months.npy" \
  --era5-dir "${REPOSITORY}/inputs/EU_74_-48_69_20/raw/ERA5-Land" \
  --input-dir "${REPOSITORY}/inputs/EU_74_-48_69_20" \
  --output-root "$STATISTICS_ROOT" \
  --run-name "$RUN_NAME"
