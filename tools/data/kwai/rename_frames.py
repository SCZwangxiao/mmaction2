import re
import os
import os.path as osp
import glob
import time
import argparse
import traceback
from tqdm import tqdm
import multiprocessing as mp


class VideoRenamer:

    class _StopToken:
        pass

    class _RenamerWorker(mp.Process):

        def __init__(self,
                     task_queue,
                     result_queue):
            self.task_queue = task_queue
            self.result_queue = result_queue
            super().__init__()

        def __rename__(self, file):
            os.system(f'bash {file}')

        def run(self):
            while True:
                task = self.task_queue.get()
                if isinstance(task, VideoRenamer._StopToken):
                    break
                file = task
                try:
                    self.__rename__(file)
                    status = True
                except Exception as e:
                    print(e)
                    traceback.print_exc()
                    # print(f'Error in {file}')
                    status = False
                self.result_queue.put((file, status))


def launch_resize_tasks(rename_commands, num_workers, batch_size):
    if not os.path.exists('.cache'):
        os.mkdir('.cache')
    # Launch pools
    print('Launching task pools...')
    task_queue = mp.Queue()
    result_queue = mp.Queue()
    procs = []
    for cpu_id in range(num_workers):
        procs.append(
            VideoRenamer._RenamerWorker(task_queue, result_queue))
    for p in procs:
        p.start()
    
    # Allocating resize jobs
    batch_id = 0
    batch_commands = []
    for command in rename_commands:
        batch_commands.append(command)
        if len(batch_commands) == batch_size:
            command_file = f'.cache/command_batch_{batch_id}.sh'
            with open(command_file, 'w') as F:
                for command in batch_commands:
                    F.write(f'{command}\n')
            batch_id += 1
            batch_commands = []
            task_queue.put(command_file)
    for _ in range(num_workers):
        task_queue.put(VideoRenamer._StopToken())
    
    # Pbar
    total = len(rename_commands)
    pbar = tqdm(total=total, desc='Renaming resized frames...')
    prev_size = total
    while task_queue.qsize() > 0:
        cur_size = task_queue.qsize()
        pbar.update(prev_size - cur_size)
        prev_size = cur_size
        time.sleep(1)
    pbar.close()


def process_resized_frames(raw_frames_path, num_clips, num_workers, batch_size):
    filename_tpl = osp.join(raw_frames_path, '{}/{}_{}.jpg')
    filename_pat = re.compile(
        osp.join(raw_frames_path, r'(\d+)/(\d+)_(\d+).jpg'))
    pid2frame_ids = {}
    pid_missing_frames = set()
    pid_renamed = set()
    rename_commands = []

    print('Scaning photo resized frame directories ...')
    video_frame_filenames = glob.glob(osp.join(resized_frames_root, '*/*.jpg'))

    for video_frame_filename in tqdm(
            video_frame_filenames, desc='Build photo frame index'):
        try:
            pid, pid, frame_id = filename_pat.findall(video_frame_filename)[0]
        except IndexError:
            frame_filename = video_frame_filename.split('/')[-1]
            assert 'single' in frame_filename or 'h' in frame_filename
            continue
        frame_id = int(frame_id)
        if pid in pid2frame_ids:
            pid2frame_ids[pid].append(frame_id)
        else:
            pid2frame_ids[pid] = [frame_id]

    for pid in tqdm(
            list(pid2frame_ids.keys()), desc='Building rename commands'):
        frame_ids = pid2frame_ids.pop(pid)

        if len(frame_ids) != num_clips:
            pid_missing_frames.add(pid)
            # video_frame_dir = osp.join(resized_frames_root, pid)
            # rename_commands.append(f'rm -rf {video_frame_dir}')
        else:
            frame_ids = sorted(frame_ids)
            if frame_ids[-1] == num_clips:
                pid_renamed.add(pid)
                continue
            for i, frame_id in enumerate(frame_ids):
                if frame_id != i + 1:
                    # some times frames 1 is the 1st frames
                    old_filename = filename_tpl.format(pid, pid, frame_id)
                    new_filename = filename_tpl.format(pid, pid, i + 1)
                    rename_commands.append(f'mv {old_filename} {new_filename}')

    with open('pid_with_missing_frames', 'w') as F:
        for pid in pid_missing_frames:
            F.write(f'{pid}\n')

    with open('pid_renamed', 'w') as F:
        for pid in pid_renamed:
            F.write(f'{pid}\n')
    
    print('Renaming %d resized raw frames...' % len(rename_commands))
    launch_resize_tasks(rename_commands, num_workers, batch_size)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Rename resized raw frames online')
    parser.add_argument(
        '--resized-frames-root',
        default='/home/wangxiao13/annotation/data/relevance/raw_frames',
        help='path of resized frames')
    parser.add_argument(
        '--num-clips', type=int, default=8, help='path of resized frames')
    parser.add_argument(
        '--num-workers', type=int, default=8, help='number of shell workers')
    parser.add_argument(
        '--batch-size', type=int, default=1000, help='number of shell workers')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    resized_frames_root = args.resized_frames_root
    num_clips = args.num_clips
    num_workers = args.num_workers
    batch_size = args.batch_size

    epoch = 1
    while True:
        print(f'Epoch {epoch} ... ')
        process_resized_frames(resized_frames_root, num_clips, num_workers, batch_size)
        epoch += 1
        time.sleep(60)
