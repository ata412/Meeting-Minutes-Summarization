#!/bin/bash
# Install additional Python packages into the shared venv.
# Run once from a login node or CPU job before submitting inference.
set -e

PROJECT=/lustrefs/disk/project/zz991000-zdeva/zz991021

module purge
module load cray-python/3.11.7

source "$PROJECT/venv/bin/activate"

export HF_HOME="$PROJECT/.hf_cache"
export PIP_CACHE_DIR="$PROJECT/.pip_cache"
mkdir -p "$PIP_CACHE_DIR"

echo "Installing exp01 deps …"
pip install \
    "vllm>=0.8.0" \
    "rank-bm25>=0.2.2" \
    "pythainlp>=5.0.0"

echo "Done. Verifying …"
python3 -c "import vllm, rank_bm25, pythainlp; print('vllm', vllm.__version__); print('rank_bm25 OK'); print('pythainlp', pythainlp.__version__)"
