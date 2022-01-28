# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
import os.path as osp

import torch
import numpy as np

from .base import BaseDataset
from .builder import DATASETS


@DATASETS.register_module()
class KwaiRawframeDataset(BaseDataset):
    """Rawframe dataset for Kwaishou video-tag relevance model.

    The dataset loads raw frames and apply specified transforms to return a
    dict containing the frame tensors and other information.

    The ann_file is a text file with multiple lines, and each line indicates
    the directory to frames of a video, vertical of the video and
    the label of a video, which are split with a whitespace.
    Example of a annotation file:

    .. code-block:: txt

        photo_id-1 0 14 343 511
        photo_id-1 1 121 243
        photo_id-1 1 223

    Args:
        ann_file (str): Path to the annotation file.
        pipeline (list[dict | callable]): A sequence of data transforms.
        total_clips (int): Number of extracted frames per video.
        data_prefix (str | None): Path to a directory where videos are held.
            Default: None.
        test_mode (bool): Store True when building test or validation dataset.
            Default: False.
        filename_tmpl (str): Template for each filename.
            Default: 'img_{:05}.jpg'.
        num_classes (int | None): Number of classes in the dataset.
            Default: None.
    """

    def __init__(self,
                 ann_file,
                 pipeline,
                 total_clips,
                 data_prefix=None,
                 test_mode=False,
                 filename_tmpl='{}_{}.jpg',
                 num_classes=None,
                 start_index=1):
        self.total_clips = total_clips
        self.filename_tmpl = filename_tmpl
        super().__init__(
            ann_file,
            pipeline,
            data_prefix,
            test_mode,
            True,
            num_classes,
            start_index,
            'RGB',
            sample_by_class=False,
            power=0,
            dynamic_length=False)

    def load_annotations(self):
        """Load annotation file to get video information."""
        if self.ann_file.endswith('.json'):
            return self.load_json_annotations()
        video_infos = []
        with open(self.ann_file, 'r') as fin:
            for line in fin:
                line_split = line.strip().split()
                video_info = {}
                idx = 0
                # idx for frame_dir
                photo_id = int(line_split[idx])
                idx += 1
                # chunk id
                chunk_id = int(line_split[idx])
                if self.data_prefix is not None:
                    frame_dir = osp.join(self.data_prefix, str(chunk_id))
                video_info['photo_id'] = photo_id
                video_info['frame_dir'] = frame_dir
                video_info['chunk_id'] = chunk_id
                video_info['total_frames'] = self.total_clips
                idx += 1
                # idx for label
                vertical = int(line_split[idx])
                video_info['label'] = vertical
                idx += 1
                # idx for frame_inds
                frame_inds = [int(x) for x in line_split[idx:]]
                if len(frame_inds) >= self.total_clips:
                    frame_inds = np.array(frame_inds)
                    total_frames = frame_inds.shape[0]
                    frame_inds = frame_inds[np.linspace(0,total_frames-1,self.total_clips).astype(np.int)]
                    video_info['frame_inds'] = frame_inds.astype(np.int)
                    video_infos.append(video_info)

        return video_infos

    def prepare_train_frames(self, idx):
        """Prepare the frames for training given the index."""
        results = copy.deepcopy(self.video_infos[idx])
        photo_id = results.pop('photo_id')
        results['filename_tmpl'] = self.filename_tmpl.format(photo_id, {})
        results['modality'] = self.modality
        results['start_index'] = self.start_index

        # prepare tensor in getitem
        onehot = torch.zeros(self.num_classes)
        onehot[results['label']] = 1.
        results['label'] = onehot

        return self.pipeline(results)

    def prepare_test_frames(self, idx):
        """Prepare the frames for testing given the index."""
        results = copy.deepcopy(self.video_infos[idx])
        photo_id = results.pop('photo_id')
        results['filename_tmpl'] = self.filename_tmpl.format(photo_id, {})
        results['modality'] = self.modality
        results['start_index'] = self.start_index

        # prepare tensor in getitem
        onehot = torch.zeros(self.num_classes)
        onehot[results['label']] = 1.
        results['label'] = onehot

        return self.pipeline(results)


@DATASETS.register_module()
class OldKwaiRawframeDataset(BaseDataset):
    """Rawframe dataset for Kwaishou video-tag relevance model.

    The dataset loads raw frames and apply specified transforms to return a
    dict containing the frame tensors and other information.

    The ann_file is a text file with multiple lines, and each line indicates
    the directory to frames of a video, vertical of the video and
    the label of a video, which are split with a whitespace.
    Example of a annotation file:

    .. code-block:: txt

        photo_id-1 0 14 343 511
        photo_id-1 1 121 243
        photo_id-1 1 223

    Args:
        ann_file (str): Path to the annotation file.
        pipeline (list[dict | callable]): A sequence of data transforms.
        total_clips (int): Number of extracted frames per video.
        data_prefix (str | None): Path to a directory where videos are held.
            Default: None.
        test_mode (bool): Store True when building test or validation dataset.
            Default: False.
        filename_tmpl (str): Template for each filename.
            Default: 'img_{:05}.jpg'.
        num_classes (int | None): Number of classes in the dataset.
            Default: None.
    """

    def __init__(self,
                 ann_file,
                 pipeline,
                 total_clips,
                 data_prefix=None,
                 test_mode=False,
                 filename_tmpl='{}_{}.jpg',
                 num_classes=None,
                 no_sampling=True,
                 start_index=1):
        self.total_clips = total_clips
        self.filename_tmpl = filename_tmpl
        self.no_sampling = no_sampling
        super().__init__(
            ann_file,
            pipeline,
            data_prefix,
            test_mode,
            True,
            num_classes,
            start_index,
            'RGB',
            sample_by_class=False,
            power=0,
            dynamic_length=False)

    def load_annotations(self):
        """Load annotation file to get video information."""
        if self.ann_file.endswith('.json'):
            return self.load_json_annotations()
        video_infos = []
        with open(self.ann_file, 'r') as fin:
            for line in fin:
                line_split = line.strip().split()
                video_info = {}
                idx = 0
                # idx for frame_dir
                if line_split[idx].endswith('mp4'):
                    photo_id = int(line_split[idx].split('.')[0])
                else:
                    photo_id = int(line_split[idx])
                
                if self.data_prefix is not None:
                    frame_dir = osp.join(self.data_prefix, str(photo_id))
                video_info['photo_id'] = photo_id
                video_info['frame_dir'] = frame_dir
                video_info['total_frames'] = self.total_clips
                idx += 1
                # idx for vertical
                video_info['vertical'] = int(line_split[idx])
                # idx += 1
                # idx for label[s]
                label = [int(x) for x in line_split[idx:]]
                assert label, f'missing label in line: {line}'
                assert self.num_classes is not None
                video_info['label'] = label

                if self.no_sampling:
                    try:
                        frame_files = os.listdir(frame_dir)
                    except FileNotFoundError:
                        continue
                    frame_inds = []
                    for file in frame_files:
                        idx = int(file.split('.')[0].split('_')[1])
                        frame_inds.append(idx)
                    frame_inds = np.array(frame_inds)
                    total_frames = frame_inds.shape[0]
                    frame_inds = frame_inds[np.linspace(0,total_frames-1,self.total_clips).astype(np.int)]
                    video_info['frame_inds'] = frame_inds.astype(np.int)

                video_infos.append(video_info)

        return video_infos

    def prepare_train_frames(self, idx):
        """Prepare the frames for training given the index."""
        results = copy.deepcopy(self.video_infos[idx])
        photo_id = results.pop('photo_id')
        results['filename_tmpl'] = self.filename_tmpl.format(photo_id, {})
        results['modality'] = self.modality
        results['start_index'] = self.start_index

        if self.no_sampling:
            results['clip_len'] = 1
            results['frame_interval'] = -1
            results['num_clips'] = self.total_clips

        # prepare tensor in getitem
        onehot = torch.zeros(self.num_classes)
        onehot[results['label']] = 1.
        results['label'] = onehot

        return self.pipeline(results)

    def prepare_test_frames(self, idx):
        """Prepare the frames for testing given the index."""
        results = copy.deepcopy(self.video_infos[idx])
        photo_id = results.pop('photo_id')
        results['filename_tmpl'] = self.filename_tmpl.format(photo_id, {})
        results['modality'] = self.modality
        results['start_index'] = self.start_index

        if self.no_sampling:
            results['clip_len'] = 1
            results['frame_interval'] = -1
            results['num_clips'] = self.total_clips

        # prepare tensor in getitem
        onehot = torch.zeros(self.num_classes)
        onehot[results['label']] = 1.
        results['label'] = onehot

        return self.pipeline(results)
