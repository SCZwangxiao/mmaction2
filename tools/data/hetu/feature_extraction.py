import argparse
import os
import os.path as osp
import pickle

import mmcv
import numpy as np
import torch

from mmaction.datasets.pipelines import Compose
from mmaction.datasets import build_dataloader, build_dataset
from mmaction.models import build_model
from mmcv.runner import load_checkpoint#, init_dist
from mmcv.parallel import MMDataParallel


def turn_off_pretrained(cfg):
    # recursively find all pretrained in the model config,
    # and set them None to avoid redundant pretrain steps for testing
    if 'pretrained' in cfg:
        cfg.pretrained = None

    # recursively turn off pretrained value
    for sub_cfg in cfg.values():
        if isinstance(sub_cfg, dict):
            turn_off_pretrained(sub_cfg)


def parse_args():
    parser = argparse.ArgumentParser(description='Extract Feature')
    parser.add_argument('--config', default='', help='training configs')
    parser.add_argument('--ckpt', help='checkpoint for feature extraction')
    parser.add_argument('--batch-size', type=int, help='batch size')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='pytorch',
        help='job launcher')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    cfg.data.videos_per_gpu = args.batch_size
    cfg.model.test_cfg = dict(average_clips=None, feature_extraction=True)

    # Build the model
    #init_dist(args.launcher, **cfg.dist_params)
    turn_off_pretrained(cfg.model)
    model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, args.ckpt)
    model.cfg = cfg
    model.eval()
    model.cuda()
    model = MMDataParallel(model, device_ids=[0])

    output_prefix = osp.join(cfg.data_root, '../video_feat')
    if not osp.exists(output_prefix):
        os.system(f'mkdir -p {output_prefix}')

    # build the dataloader
    for split in ['train', 'val', 'test']:
        print(f'Processing split {split} ...')
        dataset = build_dataset(cfg.data[split], dict(test_mode=True))
        bsize = cfg.data.get('videos_per_gpu', 1)
        dataloader_setting = dict(
            videos_per_gpu=cfg.data.get('videos_per_gpu', 1),
            workers_per_gpu=cfg.data.get('workers_per_gpu', 1),
            dist=False,
            shuffle=False)
        dataloader_setting = dict(dataloader_setting,
                                **cfg.data.get('test_dataloader', {}))
        data_loader = build_dataloader(dataset, **dataloader_setting)

        # enumerate Untrimmed videos, extract feature from each of them
        assert len(data_loader) == int((len(dataset)-0.5)//bsize) + 1
        prog_bar = mmcv.ProgressBar(len(dataset)//bsize)
        for i, data in enumerate(data_loader):
            try:
                infos = dataset.video_infos[i*bsize:(i+1)*bsize]
            except:
                infos = dataset.video_infos[i*bsize:-1]

            with torch.no_grad():
                feat = model(return_loss=False, **data)
            
            for i, info in enumerate(infos):
                frame_dir = info['filename']
                output_file = os.path.basename(osp.splitext(frame_dir)[0]) + '.pkl'
                output_file = osp.join(output_prefix, output_file)
                with open(output_file, 'wb') as fout:
                    pickle.dump(feat[i,:], fout)
            prog_bar.update()


if __name__ == '__main__':
    main()
