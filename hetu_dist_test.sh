#!/usr/bin/env bash

PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=8 --master_port=$PORT \
./tools/test.py \
configs/recognition/tanet/hetu/tanet_r50_video_1x1x8_100e_hetu_rgb.py \
work_dirs/tanet_r50_video_1x1x8_100e_hetu_rgb/best_mmit_mean_average_precision_epoch_84.pth \
--eval mmit_mean_average_precision top_k_precision top_k_recall \
--out work_dirs/tanet_r50_video_1x1x8_100e_hetu_rgb/test_results.json \
--launcher pytorch