#!/bin/bash -x
# author: Sami Miaari
# e-mail: s.miaari@fz-juelich.de

#SBATCH --job-name=coord_sensitivity
#SBATCH --account=jicg41
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH --mail-user=s.miaari@fz-juelich.de
#SBATCH --mail-type=ALL


set -eo pipefail
: "${SLURM_JOB_ID:?Run this script through the Slurm submission launcher}"
: "${SLURM_ARRAY_TASK_ID:?A Slurm array task must select the scenario}"
case "$SLURM_ARRAY_TASK_ID" in
    1|2|3) scenario="A$SLURM_ARRAY_TASK_ID" ;;
    *) printf 'Unknown scenario index %s\n' "$SLURM_ARRAY_TASK_ID" >&2; exit 2 ;;
esac
run="${1:?Pass the experiment run directory}"
source /p/project1/cslts/miaari1/python_scripts/DailyScriptBox/envs/loadenv.JURECA-DC_JUWELS_2025_GCC-OpenMPI_Std-Py-AI.ini
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 NCCL_SOCKET_IFNAME=ib0
# JURECA binds one GPU per Slurm task by default. torchrun needs all four.
export CUDA_VISIBLE_DEVICES=0,1,2,3
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"
python -m LSTM_model.coordinate_sensitivity.prepare verify --run "$run"

prepared="$run/prepared/$scenario"
predictions="$run/predictions/$scenario"
mkdir -p "$predictions"
exec 9>"$predictions/.prediction.lock"
flock -n 9
for ((member=0; member<100; member++)); do
    printf -v directory '%s/member_%03d' "$predictions" "$member"
    if [[ -s "$directory/daily_wtd.npy" && -s "$directory/daily_wtd.json" ]]; then
        printf 'Skipping completed %s member %03d\n' "$scenario" "$member"
        continue
    fi
    srun --ntasks=1 --gres=gpu:4 python -m torch.distributed.run \
        --standalone --nnodes=1 --nproc-per-node=4 \
        --module LSTM_model.model.predict_era5land_observations \
        --member "$member" --prepared-directory "$prepared" --output-directory "$predictions" \
        --expected-world-size 4 --atomic-member-output
done
