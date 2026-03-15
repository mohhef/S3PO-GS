#!/bin/bash
# Installation script for S3PO-GS.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ENV_NAME="S3PO-GS"
ENV_FILE="environment_fixed.yml"
CHECKPOINT_DIR="${SCRIPT_DIR}/checkpoints"
CHECKPOINT_NAME="MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
CHECKPOINT_URL="https://download.europe.naverlabs.com/ComputerVision/MASt3R/${CHECKPOINT_NAME}"
SKIP_CHECKPOINT_DOWNLOAD=0

usage() {
    cat <<'USAGE'
Usage: ./install_all.sh [--skip-checkpoint-download]

Sets up the S3PO-GS conda environment, builds CUDA extensions, and downloads
its required MASt3R checkpoint.
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-checkpoint-download)
            SKIP_CHECKPOINT_DOWNLOAD=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

load_conda() {
    if command -v conda >/dev/null 2>&1; then
        return 0
    fi

    if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
        # shellcheck disable=SC1091
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
        # shellcheck disable=SC1091
        source "$HOME/anaconda3/etc/profile.d/conda.sh"
    else
        echo "Error: conda not found. Install Miniconda or Anaconda first." >&2
        exit 1
    fi
}

download_file() {
    local url="$1"
    local target="$2"

    if command -v wget >/dev/null 2>&1; then
        wget -O "$target" "$url"
    elif command -v curl >/dev/null 2>&1; then
        curl -L "$url" -o "$target"
    else
        echo "Error: neither wget nor curl is installed." >&2
        exit 1
    fi
}

if [[ ! -f "$ENV_FILE" ]]; then
    echo "Error: expected environment file not found: $ENV_FILE" >&2
    exit 1
fi

if [[ ! -d "submodules/simple-knn" || ! -d "submodules/diff-gaussian-rasterization" ]]; then
    echo "Error: required CUDA extension directories are missing under submodules/." >&2
    exit 1
fi

if [[ ! -d "croco/models/curope" || ! -d "dust3r/croco/models/curope" ]]; then
    echo "Error: expected curope source directories are missing." >&2
    exit 1
fi

echo "=== S3PO-GS Installation ==="

load_conda
if ! command -v conda >/dev/null 2>&1; then
    echo "Error: conda command unavailable after initialization." >&2
    exit 1
fi

eval "$(conda shell.bash hook)"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "Conda environment '$ENV_NAME' already exists. Updating it from $ENV_FILE ..."
    conda env update -n "$ENV_NAME" -f "$ENV_FILE"
else
    echo "Creating conda environment '$ENV_NAME' from $ENV_FILE ..."
    conda env create -f "$ENV_FILE"
fi

conda activate "$ENV_NAME"

export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export CPATH="$CONDA_PREFIX/include:${CPATH:-}"
export LIBRARY_PATH="$CONDA_PREFIX/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

echo "CUDA_HOME set to: $CUDA_HOME"

echo "Installing Gaussian splatting CUDA extensions ..."
pip install --no-build-isolation submodules/simple-knn
pip install --no-build-isolation submodules/diff-gaussian-rasterization

echo "Building RoPE CUDA kernels ..."
(
    cd croco/models/curope
    python setup.py build_ext --inplace
)
(
    cd dust3r/croco/models/curope
    python setup.py build_ext --inplace
)

if [[ "$SKIP_CHECKPOINT_DOWNLOAD" -eq 0 ]]; then
    mkdir -p "$CHECKPOINT_DIR"
    if [[ -f "$CHECKPOINT_DIR/$CHECKPOINT_NAME" ]]; then
        echo "Checkpoint already present: $CHECKPOINT_DIR/$CHECKPOINT_NAME"
    else
        echo "Downloading MASt3R checkpoint ..."
        download_file "$CHECKPOINT_URL" "$CHECKPOINT_DIR/$CHECKPOINT_NAME"
    fi
fi

echo
echo "=== Installation Complete ==="
echo "To use S3PO-GS:"
echo "  conda activate $ENV_NAME"
echo "  CUDA_VISIBLE_DEVICES=0 python slam.py --config configs/mono/KITTI/07.yaml"
