#!/bin/bash
#SBATCH --job-name=exp01-apptainer
#SBATCH --partition=gpu
#SBATCH --account=zz991021
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=/lustrefs/disk/project/zz991000-zdeva/zz991021/logs/exp01_apptainer_%j.out
#SBATCH --error=/lustrefs/disk/project/zz991000-zdeva/zz991021/logs/exp01_apptainer_%j.err

set -e

PROJECT=/lustrefs/disk/project/zz991000-zdeva/zz991021
SIF="$PROJECT/toey/exp01/exp01.sif"
RESULT="$PROJECT/toey/exp01/result"

module load Apptainer/1.1.6

mkdir -p "$RESULT" "$PROJECT/logs"

echo "=== exp01 Apptainer test ==="
echo "SIF     : $SIF"
echo "RESULT  : $RESULT"
echo "GPU     : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'n/a')"

apptainer exec --nv --containall --pwd /model \
    --bind "$PROJECT/dataset:/model/test:ro" \
    --bind "$PROJECT/ua047/textsum/benchmark_lib:/benchmark_lib:ro" \
    --bind "$RESULT:/result" \
    --bind "$PROJECT/.hf_cache:/hf_cache:ro" \
    --env VLLM_WORKER_MULTIPROC_METHOD=spawn \
    "$SIF" python3 /model/run.py

echo "=== Done ==="
