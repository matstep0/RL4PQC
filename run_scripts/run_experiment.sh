#!/bin/bash

#SBATCH --job-name=RLjob
#SBATCH --account=project_462000795
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=2G
#SBATCH --cpus-per-task=1
#SBATCH --time=10:00:00
#SBATCH --export=CONFIG_PATH,STORE_DIR,CONTAINER   # Export custom variables


# fallback: if env var is unset or empty, use positional
CONFIG_PATH=${CONFIG_PATH:-$1}
STORE_DIR=${STORE_DIR:-$2}
CONTAINER=${CONTAINER:-$3}
#export OMP_NUM_THREADS=4 # CPU threads 


if [[ -z "$CONFIG_PATH" || -z "$STORE_DIR" || -z "$CONTAINER" ]]; then
    echo "ERROR: must set CONFIG_PATH, STORE_DIR and CONTAINER"
    echo "Set ENV variables or use positional arguments"
    exit 1
fi

START_TIME=$(date +%s)

singularity exec \
    --bind "$(pwd)":/mnt \
    "$CONTAINER" \
    python3 ./run_scripts/rloop.py \
      --config_file "$CONFIG_PATH" \
      --store_directory "$STORE_DIR"

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

printf "Execution time: %02d:%02d:%02d\n" \
    $((ELAPSED/3600)) \
    $(((ELAPSED%3600)/60)) \
    $((ELAPSED%60))
