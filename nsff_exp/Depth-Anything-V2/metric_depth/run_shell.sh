#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh

conda activate dino-tracker
export CUDA_VISIBLE_DEVICES=2
python run.py --encoder vitl --load-from checkpoints/depth_anything_v2_metric_hypersim_vitl.pth --max-depth 20 --data-dir ./data/gopro/250213/big_jump_1/dense --save-numpy


# python run.py --encoder vitl --load-from checkpoints/depth_anything_v2_metric_hypersim_vitl.pth --max-depth 10 --data-dir ./data/HDR-Hexplane-dataset/test/punch_128 --save-numpy
# python run.py --encoder vitl --load-from checkpoints/depth_anything_v2_metric_hypersim_vitl.pth --max-depth 10 --data-dir ./data/HDR-Hexplane-dataset/test/standup_128 --save-numpy

# python run.py --encoder vitl --load-from checkpoints/depth_anything_v2_metric_hypersim_vitl.pth --max-depth 20 --data-dir ./data/gopro/approach_step_3/dense --save-numpy
# python run.py --encoder vitl --load-from checkpoints/depth_anything_v2_metric_hypersim_vitl.pth --max-depth 20 --data-dir ./data/real/dslr/leg_up_2/dense/ --save-numpy
# python run.py --encoder vitl --load-from checkpoints/depth_anything_v2_metric_hypersim_vitl.pth --max-depth 20 --data-dir ./data/real/dslr/v_swing_1/dense/ --save-numpy

# python run.py --encoder vitl --load-from checkpoints/depth_anything_v2_metric_hypersim_vitl.pth --max-depth 20 --data-dir ./data/real/dslr//dense/ --save-numpy
# python run.py --encoder vitl --load-from checkpoints/depth_anything_v2_metric_hypersim_vitl.pth --max-depth 20 --data-dir ./data/real/dslr/changeup_arm_swing_3/dense/ --save-numpy


# python run.py --encoder vitl --load-from checkpoints/depth_anything_v2_metric_vkitti_vitl.pth --max-depth 20 --data-dir ./data/gopro/250213/big_jump_1/dense --save-numpy