#!/bin/bash
#SBATCH --job-name=exp01
#SBATCH --partition=gpu
#SBATCH --account=zz991021
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=01:30:00
#SBATCH --output=/lustrefs/disk/project/zz991000-zdeva/zz991021/logs/exp01_%j.out
#SBATCH --error=/lustrefs/disk/project/zz991000-zdeva/zz991021/logs/exp01_%j.err

set -e

PROJECT=/lustrefs/disk/project/zz991000-zdeva/zz991021

module purge
module load cray-python/3.11.7
module load cpe-cuda/23.09

source "$PROJECT/venv/bin/activate"

export TEST_DIR="$PROJECT/dataset"
export RESULT_DIR="$PROJECT/toey/exp01/result"
export PROGRESS_LIB="$PROJECT/textsum/benchmark_lib/progress"

# Expose libcuda.so without polluting LD_LIBRARY_PATH with all of /usr/lib64
CUDA_STUB="$PROJECT/.cuda_stub"
mkdir -p "$CUDA_STUB"
for f in /usr/lib64/libcuda.so /usr/lib64/libcuda.so.1 /usr/lib64/libcuda.so.565.57.01; do
    [ -e "$f" ] && ln -sf "$f" "$CUDA_STUB/$(basename "$f")"
done
export LD_LIBRARY_PATH="$CUDA_STUB:${LD_LIBRARY_PATH:-}"

export HF_HOME="$PROJECT/.hf_cache"
export TRANSFORMERS_CACHE="$PROJECT/.hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# vLLM uses triton/torch compile cache — put on fast Lustre
export VLLM_CACHE_ROOT="$PROJECT/.vllm_cache"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

mkdir -p "$RESULT_DIR" "$PROJECT/logs" "$VLLM_CACHE_ROOT"

echo "=== exp01: Qwen3-32B-AWQ + BM25/bge-m3 RRF ==="
echo "TEST_DIR   : $TEST_DIR"
echo "RESULT_DIR : $RESULT_DIR"
echo "GPU        : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'n/a')"

python3 - <<'PYEOF'
import ctypes, torch
print(f"torch: {torch.__version__}  cuda_compiled: {torch.version.cuda}")
print(f"LD_LIBRARY_PATH: {__import__('os').environ.get('LD_LIBRARY_PATH','(unset)')}")
try:
    lib = ctypes.CDLL("libcuda.so.1")
    v = ctypes.c_int(0)
    ret = lib.cuDriverGetVersion(ctypes.byref(v))
    print(f"cuDriverGetVersion: ret={ret}  version={v.value}")
except Exception as e:
    print(f"libcuda.so.1 load FAILED: {e}")
print(f"torch.cuda.is_available: {torch.cuda.is_available()}")
PYEOF

cd "$PROJECT/toey/exp01"
python3 run.py

echo "=== Inference done ==="
