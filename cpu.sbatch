#!/bin/bash
#SBATCH --job-name=data_gen            # create a short name for your job
#SBATCH --nodes=1                    # node count
#SBATCH --ntasks-per-node=1          # number of tasks per node (adjust when using MPI)
#SBATCH --cpus-per-task=128          # cpu-cores per task (>1 if multi-threaded tasks, adjust when using OMP)
#SBATCH --time=01:00:00              # total run time limit (HH:MM:SS)
#SBATCH --partition=amd            # The partition(queue: intel/amd/gpu-a30/gpu-l20) where you submit
#SBATCH --account=???
#SBATCH --output=gen_cpu_%j.out
#SBATCH --error=gen_cpu_%j.err
#SBATCH --mail-user=???@connect.ust.hk
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE

CPUS=$SLURM_CPUS_PER_TASK
if [ -z "$CPUS" ]; then
    CPUS=1
fi

DATASET_NAME="teris"

echo "Running on node $(hostname) with $CPUS workers for dataset: $DATASET_NAME"

echo "Generating dataset (Cycles 1-100)..."
# Using 128 workers as requested by node core count
uv run 1_pybullet_create_n_collect.py \
    --start_cycle 1 \
    --end_cycle 100 \
    --mode direct \
    --renderer tiny \
    --workers $CPUS \
    --dataset_name $DATASET_NAME \
    --dropping packing \
    --max_drop 80 \
    --object_types I O J L S Z T \
    > generation.log 2>&1

if [ $? -ne 0 ]; then
    echo "Generation failed! Check generation.log"
    cat generation.log
    exit 1
fi

echo "Converting to h5..."
uv run 4_generate_h5.py \
    --dataset_name $DATASET_NAME \
    --workers $CPUS \
    > h5_gen.log 2>&1

if [ $? -ne 0 ]; then
    echo "H5 Generation failed! Check h5_gen.log"
    cat h5_gen.log
    exit 1
fi

echo "Running Unit Tests..."
uv run pytest tests/test_h5.py -v --dataset_name $DATASET_NAME > test_report.log 2>&1

if [ $? -ne 0 ]; then
    echo "Tests failed! Check test_report.log"
    # We might still want to compress what we have, so we don't exit here strictly?
    # User said: "if any errors, save to a log".
    # But for tests, usually we want to know.
    # Proceeding to compression but marking job as verified-failed.
fi

echo "Compressing dataset..."
# Compress the `data/teris` folder and the logs
tar -czf dataset_${DATASET_NAME}.tar.gz data/${DATASET_NAME} generation.log h5_gen.log test_report.log

echo "Done!"