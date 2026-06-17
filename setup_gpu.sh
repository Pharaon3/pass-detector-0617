#!/usr/bin/env bash
# Reinstall PyTorch with CUDA support inside the active venv.
set -euo pipefail

CUDA_WHEEL="${1:-cu121}"  # cu121 | cu118 | cu124

echo "Uninstalling existing torch/torchvision..."
pip uninstall -y torch torchvision 2>/dev/null || true

case "$CUDA_WHEEL" in
  cu118)
    INDEX="https://download.pytorch.org/whl/cu118"
    ;;
  cu121)
    INDEX="https://download.pytorch.org/whl/cu121"
    ;;
  cu124)
    INDEX="https://download.pytorch.org/whl/cu124"
    ;;
  *)
    echo "Unknown wheel tag: $CUDA_WHEEL  (use cu118, cu121, or cu124)"
    exit 1
    ;;
esac

echo "Installing torch+torchvision from $INDEX ..."
pip install torch torchvision --index-url "$INDEX"

echo ""
python check_gpu.py
