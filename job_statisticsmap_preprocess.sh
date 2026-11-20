#!/bin/bash -x
# author: xxxx
# e-mail: xxxxx

#SBATCH --job-name=statsmap_preprocess
#SBATCH --account=xxxx
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=xxxx
#SBATCH --time=03:00:00
#SBATCH --output=preprocess_%j.out
#SBATCH --error=preprocess_%j.err
#SBATCH --mail-user=xxxxx
#SBATCH --mail-type=ALL

set -euo pipefail

cd xxxx
source xxxx
module load netcdf4-python

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"

STATISTICS_ROOT=xxxxx
if [[ -z "${RUN_NAME:-}" ]]; then
    RUN_NAME="run_preprocess_${SLURM_JOB_ID:-manual}_$(date +%Y%m%d_%H%M%S)"
    RUN_DIR="${STATISTICS_ROOT}/${RUN_NAME}"
    mkdir -p "${STATISTICS_ROOT}/logs"
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
srun python -u -m LSTM_model.data.prepare_era5land_statisticsmap \
  --mask-path xxxx \
  --era5-dir xxxx \
  --input-dir xxxx \
  --output-root xxxx \
  --run-name "$RUN_NAME"
