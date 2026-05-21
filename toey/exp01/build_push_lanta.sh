#!/bin/bash
# Build and push exp01 Docker image from LANTA using podman.
# Run from the login node. No internet needed (weights copied from .hf_cache).
set -e

PROJECT=/lustrefs/disk/project/zz991000-zdeva/zz991021
IMAGE=registry.ai.in.th/2026-textsum/47b13a1c/ata412.itfm:exp01

echo "=== Login to registry ==="
podman login registry.ai.in.th -u ata412.itfm

PODMAN_ROOT="/tmp/${USER}-podman"
mkdir -p "$PODMAN_ROOT"

echo "=== Building image (build context = $PROJECT) ==="
podman --root "$PODMAN_ROOT/root" --runroot "$PODMAN_ROOT/run" \
    build \
    -f "$PROJECT/toey/exp01/Dockerfile.lanta" \
    -t "$IMAGE" \
    "$PROJECT"

echo "=== Pushing $IMAGE ==="
podman --root "$PODMAN_ROOT/root" --runroot "$PODMAN_ROOT/run" \
    push "$IMAGE"

echo "=== Cleaning up tmp storage ==="
rm -rf "$PODMAN_ROOT"

echo "=== Done ==="
