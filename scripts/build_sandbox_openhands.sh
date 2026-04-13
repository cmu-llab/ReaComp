#!/usr/bin/env bash
# Build the Apptainer sandbox SIF for openhands_agents baselines.
# No Docker required — uses apptainer build --fakeroot, falls back to reusing openhands.sif.
# Run from the project root.
set -euo pipefail

SIF_DIR=/scratch/$USER/sif_images

bash openhands_agents/scripts/build_sandbox.sh "$SIF_DIR"
