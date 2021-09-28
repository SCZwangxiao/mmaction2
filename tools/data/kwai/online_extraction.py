import argparse
import glob
import os
import os.path as osp
import time

import mmcv
import numpy as np
import torch
import torch.multiprocessing as mp
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


def extract_video_feature_single_gpu(gpu_id, config, batch_size, data_root,
                                     video_root, use_pretrain, ckpt):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    cfg = mmcv.Config.fromfile(config)
    cfg.data.videos_per_gpu = batch_size
    cfg.data['test']['ann_file'] = osp.join(data_root, f'task_gpu{gpu_id}.txt')
    cfg.data['test']['data_prefix'] = video_root
    cfg.model.test_cfg = dict(average_clips=None, feature_extraction=True)

    # Build the model
    if use_pretrain:
        model = build_model(
            cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
    else:
        turn_off_pretrained(cfg.model)
        model = build_model(
            cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
        load_checkpoint(model, ckpt)
    model.cfg = cfg
    model.eval()
    model.cuda()
    model = MMDataParallel(model, device_ids=[gpu_id])

    output_prefix = osp.join(cfg.data_root, '../video_feat')
    if not osp.exists(output_prefix):
        os.system(f'mkdir -p {output_prefix}')

    # build the dataloader
    dataset = build_dataset(cfg.data['test'], dict(test_mode=True))
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
            output_file = os.path.basename(osp.splitext(frame_dir)[0]) + '.npy'
            output_file = osp.join(output_prefix, output_file)
            np.save(output_file, feat[i])
        prog_bar.update()


def generate_dataset(gpu_id, num_gpus, data_root, video_root):
    video_files = glob.glob(osp.join(video_root, '*.mp4'))
    selected_pids = []
    for video_file in video_files:
        pid = int(video_file.split('/')[-1].split('.')[0])
        if abs(hash(pid)) % num_gpus == gpu_id:
            selected_pids.append(pid)
    with open(osp.join(data_root, f'task_gpu{gpu_id}.txt'), 'w') as F:
        for pid in selected_pids:
            F.write(f'{pid}.mp4\t{0}\n')
    print('%d videos detected, %d are sent to gpu %d' %
          (len(video_files), len(selected_pids), gpu_id))
    with open(osp.join(data_root, f'clean_gpu{gpu_id}.sh'), 'w') as F:
        for pid in selected_pids:
            video_file = osp.join(video_root, f'{pid}.mp4')
            F.write(f'rm {video_file}\n')


def delete_videos(gpu_id, data_root):
    command_file = osp.join(data_root, f'clean_gpu{gpu_id}.sh')
    os.system(f'bash {command_file}')


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
    parser.add_argument('--num-gpu', type=int, help='number gpus')
    parser.add_argument(
        '--check-duration', type=int, default=5, help='check duration')
    parser.add_argument('--data-root', type=str, help='dataset root')
    parser.add_argument('--video-root', type=str, help='dataset videos root')
    parser.add_argument(
        '--clean', action='store_true', help='dataset videos root')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='pytorch',
        help='job launcher')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    mp.set_start_method('spawn')
    args = parse_args()
    num_gpus = args.num_gpu
    check_duration = args.check_duration

    procs = [None for _ in range(num_gpus)]
    while True:
        for gpu_id, p in enumerate(procs):
            if p is None or not p.is_alive():  # Job finished
                if p:
                    p.join()
                    print('Job in gpu %d finished! Starting new job...' %
                          gpu_id)
                    if args.clean:
                        delete_videos(gpu_id, args.data_root)
                generate_dataset(gpu_id, num_gpus, args.data_root,
                                 args.video_root)
                p = mp.Process(
                    target=extract_video_feature_single_gpu,
                    args=(gpu_id, args.config, args.batch_size, args.data_root,
                          args.video_root, args.use_pretrain, args.ckpt))
                p.start()
                procs[gpu_id] = p
            else:  # Job not finished
                pass
        time.sleep(check_duration)
