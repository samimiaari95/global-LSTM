#!/bin/bash -x
# author: Sami Miaari
# e-mail: s.miaari@fz-juelich.de

#SBATCH --job-name=conus_wtd
#SBATCH --account=jicg41
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --partition=dc-gpu
#SBATCH --time=05:00:00
#SBATCH --gres=gpu:4
#SBATCH --array=0-99%10
#SBATCH --output=outputs/CONUS/era5land_grid/logs/member_%A_%a.out
#SBATCH --error=outputs/CONUS/era5land_grid/logs/member_%A_%a.err
#SBATCH --mail-user=s.miaari@fz-juelich.de
#SBATCH --mail-type=ALL

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the global-LSTM repository}"
source /p/project1/cslts/miaari1/python_scripts/DailyScriptBox/envs/loadenv.JURECA-DC_JUWELS_2025_GCC-OpenMPI_Std-Py-AI.ini
set -u

export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NCCL_SOCKET_IFNAME=ib0
prepared_directory="${1:-inputs/CONUS/monthly_anomaly_run/era5land_grid/prepared}"
prediction_directory="${2:-outputs/CONUS/era5land_grid/predictions}"

srun env -u CUDA_VISIBLE_DEVICES python -m torch.distributed.run \
    --standalone --nnodes=1 --nproc-per-node=4 \
    --module LSTM_model.data.predict_era5land_observations \
    --member "${SLURM_ARRAY_TASK_ID:?A Slurm array member is required}" \
    --prepared-directory "$prepared_directory" \
    --output-directory "$prediction_directory" \
    --expected-world-size 4
