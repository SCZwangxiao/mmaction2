#!/usr/bin/env bash

PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=8 --master_port=$PORT \
./tools/test.py \
configs/recognition/tdn/tdn_r50_video_320p_5x1x8_100e_kinetics400_rgb.py \
work_dirs/tdn_r50_video_320p_5x1x8_100e_kinetics400_rgb/epoch_100.pth \
--eval mmit_mean_average_precision top_k_precision top_k_recall \
--out work_dirs/tdn_r50_video_320p_5x1x8_100e_kinetics400_rgb/test_results.json \
--launcher pytorch
