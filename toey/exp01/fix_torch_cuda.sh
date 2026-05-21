#!/bin/bash
# Fix: torch was installed as cu130 (CUDA 13.0) but LANTA driver only supports CUDA 12.7.
# Reinstall torch from the cu126 wheel index so the runtime matches the driver.
# Run once from a login node.
set -e

PROJECT=/lustrefs/disk/project/zz991000-zdeva/zz991021

module purge
module load cray-python/3.11.7

source "$PROJECT/venv/bin/activate"

export PIP_CACHE_DIR="$PROJECT/.pip_cache"
mkdir -p "$PIP_CACHE_DIR"

echo "Current torch: $(python3 -c 'import torch; print(torch.__version__, "cuda_compiled:", torch.version.cuda)')"
echo ""

echo "Installing torch from cu126 wheel index …"
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu126 \
    --upgrade

echo ""
echo "New torch: $(python3 -c 'import torch; print(torch.__version__, "cuda_compiled:", torch.version.cuda)')"
echo ""

echo "Verifying vllm still imports …"
python3 -c "import vllm; print('vllm', vllm.__version__)"
echo ""

echo "Done. If cu126 torch version is too old for vllm, run:"
echo "  pip install torch --index-url https://download.pytorch.org/whl/cu124 --upgrade"
echo "  # CUDA 12.4 is also compatible with the 12.7 driver"
