import argparse
import os
import os.path as osp

import mmcv
import numpy as np
import torch
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint  # , init_dist

from mmaction.datasets import build_dataloader, build_dataset
from mmaction.models import build_model


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
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--ckpt', help='checkpoint for feature extraction')
    group.add_argument(
        '--use-pretrain',
        action='store_true',
        help='use pretrain model only, be cautious for some models \
        have randomized params!')
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
    # init_dist(args.launcher, **cfg.dist_params)
    if args.use_pretrain:
        model = build_model(
            cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
    else:
        turn_off_pretrained(cfg.model)
        model = build_model(
            cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
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
        assert len(data_loader) == int((len(dataset) - 0.5) // bsize) + 1
        prog_bar = mmcv.ProgressBar(len(dataset) // bsize)
        for i, data in enumerate(data_loader):
            try:
                infos = dataset.video_infos[i * bsize:(i + 1) * bsize]
            except IndexError:
                infos = dataset.video_infos[i * bsize:-1]
            # imgs = data['imgs'] # [bsize, nseg, C, clip_len, H, W]
            # nseg = imgs.shape[1]
            # imgs = imgs.reshape((-1,) + imgs.shape[2:])
            # # [bsize*nseg, C, clip_len, H, W]
            # imgs = imgs.transpose(1, 2)
            # # [bsize*nseg, clip_len, C, H, W]
            imgs = data['imgs']  # [bsize, nseg, C, H, W]
            nseg = imgs.shape[1]
            imgs = imgs.reshape((-1, ) + imgs.shape[2:])
            # [bsize*nseg, C, H, W]
            imgs = imgs.unsqueeze(1)
            # [bsize*nseg, 1, C, H, W]

            with torch.no_grad():
                feat = model(imgs, return_loss=False)
                # [bsize*nseg, feat_dim]
                feat = feat.reshape(bsize, nseg, -1)
                # [bsize, nseg, feat_dim]

            for i, info in enumerate(infos):
                frame_dir = info['filename']
                output_file = os.path.basename(
                    osp.splitext(frame_dir)[0]) + '.npy'
                output_file = osp.join(output_prefix, output_file)
                np.save(output_file, feat[i])
            prog_bar.update()


if __name__ == '__main__':
    main()
