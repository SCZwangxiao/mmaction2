import os
import json
import time
import pickle
import argparse
import os.path as osp
from tqdm import tqdm


def monitor(raw_frames_path,
            resized_frames_path,
            num_clips,
            chunk_count,
            cycle=1800):
    epoch = 0
    while True:
        epoch += 1
        print('Epoch %d...' % epoch)
        print('    Loading logs')
        with open('pid2log.pkl', 'rb') as F:
            pid2log = pickle.load(F)
        
        new_frames_cnt = 0
        print('    Scaning unresized frames...')
        for chunk in range(chunk_count):
            if osp.exists(osp.join(raw_frames_path, str(chunk))):
                raw_frame_files = os.listdir(osp.join(raw_frames_path, str(chunk)))
                for file in raw_frame_files:
                    pid, frame_id = file.split('.')[0].split('_')
                    if frame_id not in pid2log[pid]['unresized']:
                        pid2log[pid]['unresized'].add(frame_id)
                        new_frames_cnt += 1
                        pid2log[pid]['status'] = 1
        print('    %d new raw frames' % new_frames_cnt)
        
        new_resized_frames_cnt = 0
        print('    Scaning resized frames...')
        for chunk in range(chunk_count):
            if osp.exists(osp.join(resized_frames_path, str(chunk))):
                resized_frame_files = os.listdir(osp.join(resized_frames_path, str(chunk)))
                for file in resized_frame_files:
                    pid, frame_id = file.split('.')[0].split('_')
                    if frame_id not in pid2log[pid]['resized']:
                        pid2log[pid]['unresized'].discard(frame_id)
                        pid2log[pid]['resized'].add(frame_id)
                        new_resized_frames_cnt += 1
                        if frame_id == 'single':
                            pid2log[pid]['status'] = 3
                        elif frame_id == 'h':
                            pid2log[pid]['status'] = 4
                        else:
                            if len(pid2log[pid]['resized']) == num_clips:
                                pid2log[pid]['status'] = 2
        print('    %d new resized frames' % new_resized_frames_cnt)
        
        status2cnt = {
            0: 0,
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            -1: 0}
        resized_photo_cnt = 0
        reamining_frames_cnt = 0
        print('    Analyzing log...')
        for pid, log in pid2log.items():
            if pid != 'raw_frames_file_format_error':
                status2cnt[log['status']] += 1
                if log['status'] >= 2:
                    resized_photo_cnt += 1
                elif log['status'] == 1:
                    resized_photo_cnt += len(log['resized']) / num_clips
                    reamining_frames_cnt += len(log['unresized'])
        reamining_frames_cnt += status2cnt[0] * num_clips
        print('    %.1f mean resized photos' % resized_photo_cnt)

        print('    %d unseen photos' % status2cnt[0])
        print('    %d resizing photos' % status2cnt[1])
        print('    %d all resized photos' % status2cnt[2])
        print('    %d single photos' % status2cnt[3])
        print('    %d cover photos' % status2cnt[4])
        print('    %d error photos' % status2cnt[-1])
        if new_frames_cnt > 0:
            speed = new_frames_cnt / cycle
            eta = status2cnt[0]*8 / speed / 3600
            print('    Download speed %.1f frames/s, eta: %.2f h' % (speed, eta))
        else:
            print('    No new downloaded frames!')
        if new_resized_frames_cnt > 0:
            speed = new_resized_frames_cnt / cycle
            eta = reamining_frames_cnt / speed / 3600
            print('    Resizing speed %.1f frames/s, eta: %.2f h' % (speed, eta))
        else:
            print('    No new resized frames!')
        
        print('    Dumping logs...\n')
        with open('pid2log.pkl', 'wb') as F:
            pickle.dump(pid2log, F)
        time.sleep(cycle)


def create_task_result_log(task_file_path, resume):
    if not resume or not os.path.exists('pid2log_original.pkl'):
        pid2feature = {}
        print('Loading all photo_ids...')
        with open(task_file_path, 'r') as F:
            lines = F.readlines()
        for line in lines:
            photo_id = line.strip()
            pid2feature[photo_id] = {
                'unresized': set(),
                'resized': set(),
                'failed': [],
                'status': 0,
                'error': []
            }
        pid2feature['raw_frames_file_format_error'] = []
        if not os.path.exists('pid2log_original.pkl'):
            with open('pid2log_original.pkl', 'wb') as F:
                pickle.dump(pid2feature, F)
        with open('pid2log.pkl', 'wb') as F:
            pickle.dump(pid2feature, F)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Resize extracted raw frames online')
    parser.add_argument(
        '--chunk-count',
        default=1000,
        help='num files of MMU frame extractor')
    parser.add_argument(
        '--num-clips',
        default=8,
        help='num files of MMU frame extractor')
    parser.add_argument(
        '--raw-frames-path',
        default='/home/wangxiao13/dataset/download_video/unresized_frames_hetu',
        help='path to MMU extraced frames')
    parser.add_argument(
        '--resized-frames-path',
        default='/home/wangxiao13/annotation/data/hetu_human/frames',
        help='path to store resized frames')
    parser.add_argument(
        '--task-file-path',
        default='/home/wangxiao13/dataset/download_video/task.txt',
        help='path to store task photo ids')
    parser.add_argument(
        '--resume',
        action='store_false',
        help='frequency of monitoring (in seconds)')
    parser.add_argument(
        '--cycle',
        default=600,
        help='frequency of monitoring (in seconds)')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    # paths
    raw_frames_path = args.raw_frames_path
    resized_frames_path = args.resized_frames_path
    num_clips = args.num_clips
    chunk_count = args.chunk_count
    task_file_path = args.task_file_path
    resume = args.resume
    cycle = args.cycle

    create_task_result_log(task_file_path, resume)
    
    monitor(raw_frames_path,
            resized_frames_path,
            num_clips,
            chunk_count,
            cycle)