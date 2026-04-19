#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh

# Set the data path environment variable
export CUDA_VISIBLE_DEVICES=0

export config_file="/home/shindy/projects/hdr-4d/hdr-nsff/nsff_exp/configs/gopro/i2/l1/push/config_big_jump_1_i2_l1_tf_hdr_260413.txt"
python run_nerf.py --config $config_file
python evaluation_gopro.py --config $config_file
python run_nerf.py --config $config_file --render_lockcam --target_idx 2
python run_nerf.py --config $config_file --render_bt --target_idx 20 --render_tm wb 
python run_nerf.py --config $config_file --rrender_lockcam_slowmo --target_idx 2 --render_tm wb 
python evaluation_gopro_time_36.py --config $config_file

echo "All commands done"
