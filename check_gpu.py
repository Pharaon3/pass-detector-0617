#!/usr/bin/env python3
"""Diagnose GPU / CUDA setup and print a fix if PyTorch cannot use the GPU."""

from __future__ import annotations

import shutil
import subprocess
import sys


def run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def main() -> int:
    print("=" * 60)
    print("GPU / CUDA diagnostic")
    print("=" * 60)

    if shutil.which("nvidia-smi"):
        print("\n[nvidia-smi]")
        print(run(["nvidia-smi"]))
    else:
        print("\n[nvidia-smi] NOT FOUND — install NVIDIA drivers first.")
        print("  Ubuntu: sudo apt install nvidia-driver-535  (or newer)")
        print("  Then reboot: sudo reboot")

    print("\n[PyTorch]")
    import torch

    print(f"  torch version     : {torch.__version__}")
    print(f"  torch.version.cuda: {torch.version.cuda}")
    print(f"  cuda available    : {torch.cuda.is_available()}")
    print(f"  device count      : {torch.cuda.device_count()}")

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}             : {props.name} ({props.total_memory / 1024**3:.1f} GB)")
        try:
            x = torch.randn(2, 2, device="cuda")
            y = x @ x
            print(f"  GPU tensor test   : OK ({y.device})")
        except Exception as exc:
            print(f"  GPU tensor test   : FAILED — {exc}")
            print_reinstall_hint()
            return 1
        print("\nGPU is ready. Run:  python train.py --require-gpu")
        return 0

    print("\n  CUDA NOT available to PyTorch.")
    print("  Common cause: pip installed CPU-only torch, or CUDA wheel mismatches driver.")

    # Surface the internal driver version PyTorch sees (if any)
    try:
        import torch._C as _C

        if hasattr(_C, "_cuda_getDriverVersion"):
            drv = _C._cuda_getDriverVersion()
            print(f"  driver API version: {drv}")
    except Exception:
        pass

    print_reinstall_hint()
    return 1


def print_reinstall_hint() -> None:
    print("\n" + "=" * 60)
    print("FIX — reinstall PyTorch with a CUDA build matching your driver")
    print("=" * 60)
    print("""
1) Activate your venv:
     source venv/bin/activate

2) Remove the current (CPU or mismatched) install:
     pip uninstall -y torch torchvision

3) Install a CUDA wheel (pick ONE):

   CUDA 12.1 (works on most recent drivers):
     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

   CUDA 11.8 (older GPUs / drivers):
     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

   Or run the helper script:
     bash setup_gpu.sh

4) Verify:
     python check_gpu.py

5) Train on GPU:
     python train.py --require-gpu

If nvidia-smi works but PyTorch still fails, update the driver:
  https://www.nvidia.com/Download/index.aspx
""")


if __name__ == "__main__":
    sys.exit(main())
