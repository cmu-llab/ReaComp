#!/usr/bin/env bash
# Build the Apptainer sandbox SIF from the Dockerfile.
#
# Usage:
#   bash openhands_agents/scripts/build_sandbox.sh [SIF_DIR]
#
# SIF_DIR defaults to /scratch/$USER/sif_images
set -euo pipefail

SIF_DIR="${1:-/scratch/$USER/sif_images}"
SIF_PATH="$SIF_DIR/sandbox.sif"
DOCKERFILE_DIR="$(cd "$(dirname "$0")/.." && pwd)"

mkdir -p "$SIF_DIR"

echo "==> Building Docker image from $DOCKERFILE_DIR/Dockerfile..."
docker build -t oh-sandbox:latest "$DOCKERFILE_DIR"

echo "==> Saving Docker image to tarball..."
TMPTAR="$(mktemp /tmp/oh_sandbox_XXXX.tar)"
docker save oh-sandbox:latest -o "$TMPTAR"

echo "==> Building Apptainer SIF → $SIF_PATH ..."
apptainer build "$SIF_PATH" "docker-archive://$TMPTAR"
rm -f "$TMPTAR"

echo "==> Done: $SIF_PATH"
