#!/usr/bin/env bash

PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=8 --master_port=$PORT \
./tools/test.py \
configs/recognition/timesformer/hetu/timesformer_video_divST_8x32x1_15e_hetu_rgb.py \
work_dirs/timesformer_video_divST_8x32x1_15e_hetu_rgb/epoch_13.pth \
--eval mmit_mean_average_precision top_k_precision top_k_recall \
--out work_dirs/tanet_r50_video_1x1x8_100e_hetu_rgb/test_results.json \
--launcher pytorch
