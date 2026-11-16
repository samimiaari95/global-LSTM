#!/bin/bash -x
# author: Sami Miaari
# e-mail: s.miaari@fz-juelich.de

#SBATCH --job-name=era5wtdobspixels
#SBATCH --account=jicg41
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --output=/p/project1/cslts/miaari1/python_scripts/global_TL/spatio-temporal-LSTM/outputs/EU_74_-48_69_20/validation_ERA5/obspixelsera5wtdlogs/era5land-wtd-out.%j
#SBATCH --error=/p/project1/cslts/miaari1/python_scripts/global_TL/spatio-temporal-LSTM/outputs/EU_74_-48_69_20/validation_ERA5/obspixelsera5wtdlogs/era5land-wtd-err.%j
#SBATCH --gres=gpu:4
#SBATCH --partition=dc-gpu-devel
#SBATCH --time=2:00:00
#SBATCH --mail-user=s.miaari@fz-juelich.de
#SBATCH --mail-type=ALL

set -euo pipefail

cd /p/project1/cslts/miaari1/python_scripts/global_TL/spatio-temporal-LSTM
source /p/project1/cslts/miaari1/python_scripts/DailyScriptBox/envs/loadenv.JURECA-DCadapted_JUWELS_2025_GCC-OpenMPI_Std-Py-AI.ini

if [[ $# -ne 1 ]]; then
    echo "Usage: sbatch $0 <ensemble-member-0-to-99>" >&2
    exit 1
fi

ens_member="$1"
if [[ ! "$ens_member" =~ ^[0-9]{1,2}$ ]]; then
    echo "Error: ensemble member must be an integer from 0 to 99." >&2
    exit 1
fi
export ENS_MEMBER="$ens_member"
echo "Selected ensemble member: $ENS_MEMBER"

export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
if [[ "${SYSTEMNAME:-}" == "juwelsbooster" \
    || "${SYSTEMNAME:-}" == "juwels" \
    || "${SYSTEMNAME:-}" == "jurecadc" \
    || "${SYSTEMNAME:-}" == "jusuf" ]]; then
    MASTER_ADDR="${MASTER_ADDR}i"
fi
export MASTER_PORT=54123
export NCCL_SOCKET_IFNAME=ib0

srun env -u CUDA_VISIBLE_DEVICES python -u -m torchrun_jsc \
    --nproc_per_node=4 \
    --nnodes="$SLURM_JOB_NUM_NODES" \
    --rdzv_id="$SLURM_JOB_ID" \
    --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    --rdzv_backend=static \
    -m LSTM_model.data.predict_era5land_observations \
    --member "$ens_member" \
    --expected-world-size 4
