#!/bin/bash -x
# author: Sami Miaari
# e-mail: s.miaari@fz-juelich.de

#SBATCH --job-name=era5wtdobspixels
#SBATCH --account=jicg41
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:4
#SBATCH --partition=dc-gpu
#SBATCH --time=05:00:00
#SBATCH --mail-user=s.miaari@fz-juelich.de
#SBATCH --mail-type=ALL

set -eo pipefail

if [[ -z ${SLURM_JOB_ID:-} ]]; then
    script="$(realpath "${BASH_SOURCE[0]}")"
    cd "$(dirname "$script")/../.."
    logs=outputs/EU_74_-48_69_20/validation_ERA5/obspixelsera5wtdlogs
    mkdir -p "$logs"
    exec sbatch "$@" --chdir="$PWD" \
        --output="$logs/era5land-wtd-out.%j" --error="$logs/era5land-wtd-err.%j" \
        "$script"
fi

if [[ ! -f LSTM_model/model/config.py ]]; then
    echo "Submit from the repository root or pass sbatch --chdir=/path/to/global-LSTM." >&2
    exit 1
fi
source /p/project1/cslts/miaari1/python_scripts/DailyScriptBox/envs/loadenv.JURECA-DCadapted_JUWELS_2025_GCC-OpenMPI_Std-Py-AI.ini
set -u

if [[ $# -ne 0 ]]; then
    echo "Usage: sbatch $0" >&2
    exit 1
fi

export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
if [[ "${SYSTEMNAME:-}" == "juwelsbooster" \
    || "${SYSTEMNAME:-}" == "juwels" \
    || "${SYSTEMNAME:-}" == "jurecadc" \
    || "${SYSTEMNAME:-}" == "jusuf" ]]; then
    MASTER_ADDR="${MASTER_ADDR}i"
fi
export MASTER_PORT=54123
export NCCL_SOCKET_IFNAME=ib0

for ens_member in {0..99}; do
    export ENS_MEMBER="$ens_member"
    echo "[$(date '+%F %T')] Starting ensemble member $ENS_MEMBER"
    srun env -u CUDA_VISIBLE_DEVICES python -u -m torchrun_jsc \
        --nproc_per_node=4 \
        --nnodes="$SLURM_JOB_NUM_NODES" \
        --rdzv_id="${SLURM_JOB_ID}_${ens_member}" \
        --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
        --rdzv_backend=static \
        -m LSTM_model.model.predict_era5land_observations \
        --member "$ens_member" \
        --expected-world-size 4
    echo "[$(date '+%F %T')] Finished ensemble member $ENS_MEMBER"
done

echo "[$(date '+%F %T')] Completed all 100 ensemble members"
