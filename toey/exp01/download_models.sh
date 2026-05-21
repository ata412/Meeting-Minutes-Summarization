#!/bin/bash
# Download Qwen3-32B-AWQ (and bge-m3 if not present).
# Run once from a login node — no GPU required.
set -e

PROJECT=/lustrefs/disk/project/zz991000-zdeva/zz991021

module purge
module load cray-python/3.11.7

source "$PROJECT/venv/bin/activate"

export HF_HOME="$PROJECT/.hf_cache"
export HF_HUB_OFFLINE=0

echo "Downloading BAAI/bge-m3 (skip if cached) …"
python3 - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download("BAAI/bge-m3")
print("bge-m3 ready")
EOF

echo "Downloading Qwen/Qwen3-32B-AWQ (~18 GB) …"
python3 - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen3-32B-AWQ")
print("Qwen3-32B-AWQ ready")
EOF

echo "All models downloaded to $HF_HOME"
