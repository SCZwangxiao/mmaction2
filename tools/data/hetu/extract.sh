#PYTHONPATH="$(dirname $0)/..":$PYTHONPATH python -m torch.distributed.launch --nproc_per_node=1 --master_port=29500 \
python tools/data/hetu/feature_extraction.py --config configs/recognition/timesformer/hetu/timesformer_video_divST_8x32x1_15e_hetu_rgb.py --ckpt \
work_dirs/timesformer_video_divST_8x32x1_15e_hetu_rgb/epoch_13.pth --batch-size 2