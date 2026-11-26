#!/bin/bash -x
# author: Sami Miaari
# e-mail: s.miaari@fz-juelich.de

#SBATCH --job-name=conusmonthly_wtd
#SBATCH --account=jicg41
#SBATCH --partition=dc-gpu-devel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:4
#SBATCH --time=00:45:00
#SBATCH --mail-user=s.miaari@fz-juelich.de
#SBATCH --mail-type=ALL

set -eo pipefail
prepared=inputs/CONUSmonthly/era5land_grid/prepared
output=outputs/CONUSmonthly
predictions="$output/predictions"

if [[ -z ${SLURM_JOB_ID:-} ]]; then
    script="$(realpath "${BASH_SOURCE[0]}")"
    cd "$(dirname "$script")/../.."
    test -f "$prepared/preprocessing.json"
    mkdir -p "$output/logs"
    exec sbatch "$@" --clusters=jureca --chdir="$PWD" \
        --output="$output/logs/prediction_%j.out" --error="$output/logs/prediction_%j.err" \
        "$script"
fi

if [[ ! -f LSTM_model/model/config.py || ! -f "$prepared/preprocessing.json" ]]; then
    printf 'Missing %s/%s/preprocessing.json. Submit from the repository root.\n' "$PWD" "$prepared" >&2
    exit 1
fi
source /p/project1/cslts/miaari1/python_scripts/DailyScriptBox/envs/loadenv.JURECA-DC_JUWELS_2025_GCC-OpenMPI_Std-Py-AI.ini
set -u
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NCCL_SOCKET_IFNAME=ib0

mkdir -p "$predictions"
exec 9>"$predictions/.prediction.lock"
if ! flock -n 9; then
    printf 'A prediction job is already using %s\n' "$predictions" >&2
    exit 1
fi

for ((member=0; member<100; member++)); do
    printf -v member_directory '%s/member_%03d' "$predictions" "$member"
    if [[ -s "$member_directory/daily_wtd.npy" && -s "$member_directory/daily_wtd.json" ]]; then
        printf 'Member %03d already has output, skipping\n' "$member"
        continue
    fi
    printf 'Running ensemble member %03d\n' "$member"
    srun env -u CUDA_VISIBLE_DEVICES python -m torch.distributed.run \
        --standalone --nnodes=1 --nproc-per-node=4 \
        --module LSTM_model.model.predict_era5land_observations \
        --member "$member" \
        --prepared-directory "$prepared" \
        --output-directory "$predictions" \
        --expected-world-size 4
done
