#!/usr/bin/env zsh
set -e
source ~/anaconda3/etc/profile.d/conda.sh
conda activate dino-tracker

# ── Data path ──────────────────────────────────────────────────────────────
export data_path="/home/shindy/projects/hdr-4d/hdr-nsff/data/hdr-gopro/bear_thread_test/dense"

SCRIPT_DIR="$(cd "$(dirname "${(%):-%x}")" && pwd)"
DINO_DIR="${SCRIPT_DIR}/dino-tracker"

export CUDA_VISIBLE_DEVICES=2

# ── Set up dino-tracker input directories ──────────────────────────────────
mkdir -p "$data_path/dino-tracker"
rm -f "$data_path/dino-tracker/video"
rm -f "$data_path/dino-tracker/masks"
ln -s "$data_path/images"       "$data_path/dino-tracker/video"
ln -s "$data_path/motion_masks" "$data_path/dino-tracker/masks"

# ── Run DINO-tracker pipeline ──────────────────────────────────────────────
export dino_path="$data_path/dino-tracker"
export OLD_PYTHONPATH="$PYTHONPATH"
export PYTHONPATH="${DINO_DIR}:$PYTHONPATH"

cd "${DINO_DIR}"

# python ./preprocessing/main_preprocessing.py \
#     --config ./config/preprocessing.yaml \
#     --data-path "$dino_path"

# python ./train.py \
#     --config ./config/train.yaml \
#     --data-path "$dino_path"

python ./inference_semantic_flow.py \
    --config ./config/train.yaml \
    --data-path "$dino_path" \
    --interval 1

# ── Convert semantic flow npy → semantic_flow_i{step} npz ────────────────
cd "${SCRIPT_DIR}"
export PYTHONPATH="$OLD_PYTHONPATH"

python run_flows_video.py \
    --semantic_flow \
    --data_path "$data_path" \
    --skip_of \
    --skip_moseg
