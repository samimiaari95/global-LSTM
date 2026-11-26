#!/bin/bash
set -euo pipefail

cat >&2 <<'EOF'
The statistics-map stages are now submitted separately:

  1. bash scripts/statistics_maps/submit_statisticsmap_preprocess.sh [run_name]
  2. bash scripts/statistics_maps/submit_statisticsmap_prediction.sh <run_name>
  3. bash scripts/statistics_maps/submit_statisticsmap_postprocess.sh <run_name>

Job IDs are stored inside the run directory and used internally for Slurm
dependencies; you do not need to copy them between commands. The prediction
helper submits 100 separate jobs, one for each member index 0-99. Each job
requests 4 GPUs. Slurm schedules them according to account limits and GPU
availability.
EOF
exit 2
