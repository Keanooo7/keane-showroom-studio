#!/usr/bin/env bash
# setup_gpu.sh - one-time environment setup for SuGaR on an Ubuntu 20.04/22.04 + NVIDIA box
# (RunPod / Vast.ai / Lambda / GCP).  Mirrors the official steps:
#   README "Installation" (github.com/Anttwo/SuGaR, fetched 2026-10-07):
#     git clone https://github.com/Anttwo/SuGaR.git --recursive
#     python install.py           # = conda env create -f environment.yml
#     conda activate sugar        #   + pip install -e diff-gaussian-rasterization, simple-knn
#                                 #   + git clone nvdiffrast && pip install .
#   Requirements stated there: Conda, a C++ compiler, CUDA toolkit 11.8, compiler/CUDA SDK compatible.
# We run install.py's steps one by one (same commands) so we can insert the CUDA 11.8
# toolkit into the env when the host nvcc is a different version - the #1 build failure.
#
# Usage:  bash setup_gpu.sh [INSTALL_DIR=$HOME]
set -euo pipefail

INSTALL_DIR="${1:-$HOME}"
SUGAR_DIR="$INSTALL_DIR/SuGaR"
log() { printf '\n[setup %s] %s\n' "$(date +%H:%M:%S)" "$*"; }

log "GPU check"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || {
  echo "No NVIDIA GPU visible (nvidia-smi failed). Use a CUDA instance."; exit 1; }
VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if [ "${VRAM_MB:-0}" -lt 20000 ]; then
  echo "WARNING: ${VRAM_MB} MB VRAM. 24 GB is recommended (3DGS paper-quality guidance); use --low_poly / fewer, smaller frames."
fi

log "System packages (git, ffmpeg, colmap, build tools, imagemagick for convert.py --resize)"
SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"
$SUDO apt-get update -y
DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y \
  git wget ffmpeg colmap build-essential ninja-build imagemagick libgl1 libglib2.0-0
if colmap -h 2>&1 | grep -qi "with CUDA"; then
  echo "COLMAP built with CUDA."
else
  echo "NOTE: apt COLMAP has no CUDA -> run_pipeline.sh will pass --no_gpu to convert.py (slower SIFT)."
  echo "      For speed: conda install -n sugar -c conda-forge 'colmap=*=*cuda*'  (optional)."
fi

log "Miniconda (if conda missing)"
if ! command -v conda >/dev/null 2>&1; then
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
  export PATH="$HOME/miniconda3/bin:$PATH"
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >/dev/null 2>&1 || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r >/dev/null 2>&1 || true

log "Clone SuGaR (recursive: gaussian_splatting + submodules)"
if [ ! -d "$SUGAR_DIR" ]; then
  git clone https://github.com/Anttwo/SuGaR.git --recursive "$SUGAR_DIR"
else
  git -C "$SUGAR_DIR" submodule update --init --recursive
fi
cd "$SUGAR_DIR"

log "Conda env 'sugar' from environment.yml (python 3.9, pytorch 2.0.1, pytorch-cuda 11.8, pytorch3d 0.7.4)"
if ! conda env list | grep -qE '^sugar\s'; then
  conda env create -f environment.yml
fi
conda activate sugar

log "CUDA toolkit 11.8 for building the rasterizer"
if ! nvcc --version 2>/dev/null | grep -q "release 11.8"; then
  echo "Host nvcc is not 11.8 -> installing cuda-toolkit 11.8 into the env"
  conda install -y -c "nvidia/label/cuda-11.8.0" cuda-toolkit
  export CUDA_HOME="$CONDA_PREFIX"
fi
# compile for the GPU we are on (avoids 'no kernel image' errors)
CC_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
export TORCH_CUDA_ARCH_LIST="${CC_CAP}"

log "Gaussian Splatting rasterizer + simple-knn (README step b)"
pip install -e gaussian_splatting/submodules/diff-gaussian-rasterization/
pip install -e gaussian_splatting/submodules/simple-knn/

log "Nvdiffrast (README step c: texture extraction in seconds instead of minutes)"
if [ ! -d nvdiffrast ]; then git clone https://github.com/NVlabs/nvdiffrast; fi
pip install ./nvdiffrast

log "Post-processor deps (our cad_clean.py) - into the same env"
pip install "trimesh>=4" manifold3d shapely scipy rtree

log "Smoke test"
python - <<'EOF'
import torch, pytorch3d, diff_gaussian_rasterization, simple_knn  # noqa
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpu", torch.cuda.get_device_name(0))
try:
    import nvdiffrast.torch  # noqa
    print("nvdiffrast OK")
except Exception as e:
    print("nvdiffrast import failed (texture step will be slow):", e)
EOF
log "Done. SuGaR at $SUGAR_DIR  ->  next: bash run_pipeline.sh --video kitchen.mov --name job01"
