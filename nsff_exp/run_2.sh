#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh

# Set the data path environment variable
export CUDA_VISIBLE_DEVICES=2

export config_file="./config_pub/config_bear_thread.txt"
python run_nerf.py --config $config_file
python evaluation_gopro.py --config $config_file
python run_nerf.py --config $config_file --render_lockcam --target_idx 2
python run_nerf.py --config $config_file --render_bt --target_idx 20 --render_tm wb 
python run_nerf.py --config $config_file --rrender_lockcam_slowmo --target_idx 2 --render_tm wb 
python evaluation_gopro_time_36.py --config $config_file

echo "All commands done"
