#PYTHONPATH="$(dirname $0)/..":$PYTHONPATH python -m torch.distributed.launch --nproc_per_node=1 --master_port=29500 \
python tools/data/kwai/online_extraction.py \
--config tools/data/kwai/tsn_swintransformer_feature_extraction_kwai_3x1x8_rgb.py \
--use-pretrain \
--batch-size 1 \
--num-gpu 2 \
--check-duration 5 \
--data-root /home/wangxiao13/dataset/download_video \
--video-root /home/wangxiao13/dataset/download_video/video_valid \
--clean
