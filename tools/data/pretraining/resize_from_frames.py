import argparse
import glob
import pickle
import multiprocessing as mp
import os
import os.path as osp
import time
import json
import traceback

import cv2
from tqdm import tqdm


class VideoFrameExtractor:

    class _StopToken:
        pass

    class _ExtractorWorker(mp.Process):

        def __init__(self,
                     resized_frames_root,
                     task_queue,
                     result_queue,
                     short_side=256):
            self.resized_frames_root = resized_frames_root
            self.short_side = short_side
            self.task_queue = task_queue
            self.result_queue = result_queue
            super().__init__()

        def __resize__(self, file):
            # Get path
            filename = file.split('/')[-1]
            pid = filename.split('.')[0].split('_')[0]
            resized_frames_dir = osp.join(self.resized_frames_root, pid)
            try:
                os.mkdir(resized_frames_dir)
            except OSError:
                # Thread safe
                pass
            resized_frames_path = osp.join(resized_frames_dir, filename)
            # Load
            image = cv2.imread(file)
            # Resize
            h, w, _ = image.shape
            if w > h:
                h_new = short_side
                w_new = w * short_side // h
            else:
                w_new = short_side
                h_new = h * short_side // w
            image = cv2.resize(
                image, (w_new, h_new), interpolation=cv2.INTER_LINEAR)
            # Export
            cv2.imwrite(resized_frames_path, image)

        def run(self):
            """
            task_queue:
                file path: '../{pid}_{frame_id}.jpg'
            result_queue:
                file path: '../{pid}_{frame_id}.jpg'
                status (bool)
            
            resize file '../{pid}_{frame_id}.jpg' to
            '{resized_frames_root}/{pid}/{pid}_{frame_id}.jpg'
            """
            while True:
                task = self.task_queue.get()
                if isinstance(task, VideoFrameExtractor._StopToken):
                    break
                file = task
                try:
                    self.__resize__(file)
                    status = True
                except Exception as e:
                    print(e)
                    traceback.print_exc()
                    # print(f'Error in {file}')
                    status = False
                self.result_queue.put((file, status))


def process_raw_frames(short_side, 
                       num_workers, 
                       num_clips,
                       raw_frames_path,
                       raw_frames_file_cnt,
                       resized_frames_root):
    log_save_every = 180
    # Load result log
    with open('pid2feature.pkl', 'rb') as F:
        pid2feature = pickle.load(F)
    
    # Launch resize workers
    task_queue = mp.Queue()
    result_queue = mp.Queue()
    procs = []
    for cpu_id in range(num_workers):
        procs.append(
            VideoFrameExtractor._ExtractorWorker(resized_frames_root,
                                                 task_queue, result_queue,
                                                 short_side))
    for p in procs:
        p.start()

    # Load unsized raw frames
    print('Scaning unresized raw frame directories ...')
    raw_frames_files = glob.glob(osp.join(raw_frames_path, '*/*.jpg'))
    print('Filling task queue ...')
    total_tasks = 0
    for file in raw_frames_files:
        # add file to log
        try:
            photo_id, frame_id = file.split('/')[-1].split('.')[0].split('_')
            photo_id_digits = int(photo_id) # check photo_id format
            if frame_id == 'single':
                pid2feature[photo_id]['status'] = 3
            elif frame_id == 'h':
                pid2feature[photo_id]['status'] = 4
            else:
                frame_id_digits = int(frame_id) # check frame_id format
                pid2feature[photo_id]['status'] = 1
                if frame_id not in pid2feature[photo_id]['resized']:
                    pid2feature[photo_id]['unresized'].add(frame_id)
                    # Add file to queue
                    task_queue.put(file)
                    total_tasks += 1
        except:
            # invalid photo_id or frame_id format
            pid2feature['raw_frames_file_format_error'].append(file)
    for _ in range(num_workers):
        task_queue.put(VideoFrameExtractor._StopToken())

    # Monitor and illustrate progress
    pbar = tqdm(total=total_tasks, desc='Processing raw frames...')
    prev_size = total_tasks
    iter_cnt = 0
    while task_queue.qsize() > 0:
        iter_cnt += 1
        cur_size = task_queue.qsize()
        pbar.update(prev_size - cur_size)
        prev_size = cur_size
        # Update results log
        files_to_delete = update_task_result_log(pid2feature, 
                                                 result_queue, 
                                                 num_clips, 
                                                 max_updates=task_queue.qsize())
        # Save results log
        if iter_cnt % log_save_every == 0:
            save_task_result_log(pid2feature)
        # Delete processed frames
        # delete_processed_frames(files_to_delete)
        time.sleep(10)
    # Process remaining results
    if result_queue.qsize() > 0:
        files_to_delete = update_task_result_log(pid2feature, 
                                                 result_queue, 
                                                 num_clips, 
                                                 max_updates=len(raw_frames_files))
    save_task_result_log(pid2feature)
    # delete_processed_frames(files_to_delete)
    pbar.update(prev_size)
    pbar.close()
    
def update_task_result_log(pid2feature, result_queue, num_clips, max_updates=1000):
    print('Updating task result log...')
    files_to_delete = []
    updata_cnt = 0
    while updata_cnt < max_updates and result_queue.qsize() > 0:
        updata_cnt += 1
        file, status = result_queue.get()
        files_to_delete.append(file)
        filename = file.split('/')[-1]
        photo_id, frame_id = filename.split('.')[0].split('_')
        pid2feature[photo_id]['unresized'].discard(frame_id)
        if status == True:
            pid2feature[photo_id]['resized'].add(frame_id)
        else:
            pid2feature[photo_id]['failed'].append(frame_id)
            pid2feature[photo_id]['error'].append('resize error')
            pid2feature[photo_id]['status'] = -1
        if len(pid2feature[photo_id]['resized']) == num_clips:
            pid2feature[photo_id]['status'] = 2
    return files_to_delete

def save_task_result_log(pid2feature):
    print('Saving task result log...')
    with open('pid2feature.pkl', 'wb') as F:
        pickle.dump(pid2feature, F)

def delete_processed_frames(files_to_delete):
    with open('delete_processed_raw_frames.sh', 'w') as F:
        for file in files_to_delete:
            F.write(f'rm {file}\n')
    print('Deleting %d processed raw frames...' % len(files_to_delete))
    os.system('bash delete_processed_raw_frames.sh')
    
def create_task_result_log(task_file_path, resume):
    if not resume:
        assert not os.path.exists('pid2feature_original.pkl'), \
            'Error! a task log file already exists!'
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
        with open('pid2feature_original.pkl', 'wb') as F:
            pickle.dump(pid2feature, F)
        with open('pid2feature.pkl', 'wb') as F:
            pickle.dump(pid2feature, F)
    else:
        assert os.path.exists('pid2feature_original.pkl'), \
            'Error! previou log file not found!'

def parse_args():
    parser = argparse.ArgumentParser(
        description='Resize extracted raw frames online')
    parser.add_argument(
        '--resume',
        action='store_true',
        help='whether to resume from the original logs')
    parser.add_argument(
        '--task-file-path',
        default='task.txt',
        help='path to photo ids')
    parser.add_argument(
        '--raw-frames-path',
        default='/home/wangxiao13/dataset/download_video/video_rawframes',
        help='path to MMU extraced frames')
    parser.add_argument(
        '--raw-frames-file-count',
        default=10,
        help='number of files in MMU extraced frames path')
    parser.add_argument(
        '--num-clips',
        default=8,
        help='number of clips')
    parser.add_argument(
        '--resized-frames-root',
        default='/home/wangxiao13/annotation/data/pretraining/raw_frames',
        help='path to store resized frames')
    parser.add_argument(
        '--short-side', type=int, default=256, help='short side of video')
    parser.add_argument(
        '--num-workers', type=int, default=8, help='number of gpus')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    """
    device: 60 core 200GB
    frames: 100000
    
    delete speed 90 frames/s
    
    num-workers test results:
        8: resize+delete speed 60 frames/s, 1300%, 6.08GB
        16: resize+delete speed 50 frames/s, 1500%, 6.14GB
    """
    args = parse_args()
    # paths
    raw_frames_path = args.raw_frames_path
    raw_frames_file_cnt = args.raw_frames_file_count
    resized_frames_root = args.resized_frames_root
    num_clips = args.num_clips
    # resize configs
    short_side = args.short_side
    num_workers = args.num_workers
    
    create_task_result_log(args.task_file_path, args.resume)
    
    epoch = 1
    while True:
        print(f'Epoch {epoch} ... ')
        process_raw_frames(short_side, 
                           num_workers, 
                           num_clips,
                           raw_frames_path,
                           raw_frames_file_cnt,
                           resized_frames_root)
        epoch += 1
        time.sleep(1)
        break
