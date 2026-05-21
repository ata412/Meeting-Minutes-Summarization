#!/bin/bash
# vllm 0.21.0 links against libcudart.so.13 (CUDA 13.0), which requires a newer driver
# than LANTA has (max CUDA 12.7).  Downgrade to the newest vllm that targets CUDA 12.x.
# Run once from a login node.
set -e

PROJECT=/lustrefs/disk/project/zz991000-zdeva/zz991021

module purge
module load cray-python/3.11.7

source "$PROJECT/venv/bin/activate"

export PIP_CACHE_DIR="$PROJECT/.pip_cache"
mkdir -p "$PIP_CACHE_DIR"

# Try versions from newest to oldest until one links against libcudart.so.12
try_version() {
    local ver="$1"
    echo ">>> Trying vllm==$ver …"
    pip install "vllm==$ver" --force-reinstall --quiet
    local sofile
    sofile=$(find "$PROJECT/venv/lib/python3.11/site-packages/vllm" \
                  -name "_C*.so" 2>/dev/null | head -1)
    if [ -z "$sofile" ]; then
        echo "    _C.so not found — skipping"
        return 1
    fi
    local cuda_ver
    cuda_ver=$(ldd "$sofile" 2>/dev/null | grep libcudart | grep -oP 'libcudart\.so\.\K[0-9]+' | head -1)
    echo "    libcudart.so.$cuda_ver linked"
    if [ "${cuda_ver:-99}" -le 12 ]; then
        echo "    ✓ CUDA 12.x wheel — keeping vllm $ver"
        return 0
    else
        echo "    ✗ CUDA 13+ wheel — trying older version"
        return 1
    fi
}

for ver in 0.20.2 0.20.1 0.20.0 0.19.1 0.19.0 0.18.1 0.18.0; do
    if try_version "$ver"; then
        break
    fi
done

echo ""
echo "vllm installed: $(python3 -c 'import vllm; print(vllm.__version__)')"
echo "torch        : $(python3 -c 'import torch; print(torch.__version__, "cuda:", torch.version.cuda)')"

echo ""
echo "Checking Qwen3 architecture support …"
python3 - <<'EOF'
from vllm.model_executor.models import ModelRegistry
models = ModelRegistry.get_supported_archs() if hasattr(ModelRegistry, "get_supported_archs") else []
if "Qwen3ForCausalLM" in str(models) or not models:
    # Also try the direct import path
    try:
        from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM
        print("Qwen3ForCausalLM: found via direct import ✓")
    except ImportError:
        try:
            from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM
            print("Qwen3ForCausalLM: not found (only Qwen2); need newer vllm")
        except Exception:
            print("Could not check Qwen3 support")
else:
    print("Qwen3ForCausalLM:", "✓" if "Qwen3ForCausalLM" in str(models) else "✗ not found")
EOF
