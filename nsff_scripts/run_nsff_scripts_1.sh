#!/usr/bin/env zsh
source ~/anaconda3/etc/profile.d/conda.sh
export CUDA_VISIBLE_DEVICES=2

run_depth_anything() {
    local data_path=$1
    local max_depth=$2
    python ../nsff_exp/Depth-Anything-V2/metric_depth/run.py --encoder vitl \
        --load-from ../nsff_exp/Depth-Anything-V2/metric_depth/checkpoints/depth_anything_v2_metric_hypersim_vitl.pth \
        --max-depth $max_depth --data-dir $data_path --save-numpy
}

export data_path="/home/shindy/projects/hdr-4d/hdr-nsff/data/hdr-gopro/bear_thread_test/dense"
python save_poses_nerf.py --data_path $data_path
python resize_images.py --data_path $data_path --resize_height 480
run_depth_anything $data_path 20
