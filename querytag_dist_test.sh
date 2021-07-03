#!/usr/bin/env bash

PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=8 --master_port=$PORT \
./tools/test.py \
configs/recognition/tanet/querytag/tanet_r50_video_1x1x8_kinetics400_ft50e_querytag_v2_rgb.py \
work_dirs/tanet_r50_video_1x1x8_kinetics400_ft50e_querytag_v2_rgb/best_mmit_mean_average_precision_epoch_50.pth \
--eval mmit_mean_average_precision top_k_precision top_k_recall \
--out work_dirs/tanet_r50_video_1x1x8_kinetics400_ft50e_querytag_v2_rgb/test_results.json \
--launcher pytorch