#!/usr/bin/env bash
# =============================================================================
# setup_dino_tracker.sh
#
# Creates the `dino-tracker` conda environment:
#   - PyTorch 2.1.0+cu121 (required by dino-tracker)
#   - dino-tracker (semantic optical flow via DINO features)
#
# NOTE: dino-tracker requires torch==2.1.0, which is incompatible with
# SAM2 (requires torch>=2.5). Use this env separately for dino-tracker,
# and hdr-nsff for hdr-nsff training and SAM2 mask generation.
#
# Usage:
#   cd nsff_scripts/
#   bash setup_dino_tracker.sh
# =============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="dino-tracker"
PYTHON_VERSION="3.10"
TORCH_INDEX="https://download.pytorch.org/whl/cu121"

echo "================================================================"
echo " dino-tracker environment setup"
echo " ENV : $ENV_NAME  |  DIR : $SCRIPT_DIR"
echo "================================================================"

source "$(conda info --base)/etc/profile.d/conda.sh"

# ---------------------------------------------------------------------------
# 1. Create conda environment
# ---------------------------------------------------------------------------
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "[1/4] Env '${ENV_NAME}' already exists — skipping."
else
    echo "[1/4] Creating conda env '${ENV_NAME}' ..."
    conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y
fi
conda activate "${ENV_NAME}"

# ---------------------------------------------------------------------------
# 2. PyTorch 2.1.0+cu121 (pinned by dino-tracker)
# ---------------------------------------------------------------------------
echo "[2/4] Installing PyTorch 2.1.0+cu121 ..."
pip install --quiet \
    torch==2.1.0+cu121 \
    torchvision==0.16.0+cu121 \
    xformers==0.0.22.post7 \
    --index-url "${TORCH_INDEX}"

# ---------------------------------------------------------------------------
# 3. Clone dino-tracker and install dependencies
# ---------------------------------------------------------------------------
echo "[3/4] Setting up dino-tracker ..."
DINO_DIR="${SCRIPT_DIR}/dino-tracker"

# dino-tracker code is included in the repo (nsff_scripts/dino-tracker/).
# No git clone needed.

# Install deps (skip strict torch pin from requirements.txt — already installed)
pip install --quiet \
    antialiased_cnns \
    einops \
    "imageio[ffmpeg]" \
    kornia \
    matplotlib \
    mediapy \
    "numpy==1.26.4" \
    pandas \
    pillow \
    tqdm \
    pyyaml \
    "opencv-python==4.8.1.78" \
    "packaging<26" \
    scipy \
    scikit-image

# ---------------------------------------------------------------------------
# 4. Register Jupyter kernel
# ---------------------------------------------------------------------------
echo "[4/4] Registering Jupyter kernel ..."
pip install --quiet ipykernel
python -m ipykernel install --user \
    --name "${ENV_NAME}" \
    --display-name "${ENV_NAME}"

echo ""
echo "================================================================"
echo " Done!  conda activate ${ENV_NAME}"
echo ""
echo " dino-tracker repo : ${DINO_DIR}"
echo ""
echo " To run dino-tracker pipeline:"
echo "   bash nsff_scripts/run_dino-tracker_0.sh"
echo "================================================================"
