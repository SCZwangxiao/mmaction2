import argparse
import glob
import multiprocessing as mp
import os
import os.path as osp
import time
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


def process_raw_frames(short_side, num_workers, raw_frames_path,
                       resized_frames_root):
    # Launch pools
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

    # Process raw frames
    print('Scaning photo raw frame directories ...')
    raw_frames_files = glob.glob(osp.join(raw_frames_path, '*/*.jpg'))
    print('Filling task queue ...')
    for file in raw_frames_files:
        task_queue.put(file)
    for _ in range(num_workers):
        task_queue.put(VideoFrameExtractor._StopToken())

    # Pbar
    total = len(raw_frames_files)
    pbar = tqdm(total=total, desc='Processing raw frames...')
    prev_size = total
    while task_queue.qsize() > 0:
        cur_size = task_queue.qsize()
        pbar.update(prev_size - cur_size)
        prev_size = cur_size
        time.sleep(1)
    pbar.close()

    with open('delete_processed_raw_frames.sh', 'w') as F:
        for file in raw_frames_files:
            F.write(f'rm {file}\n')
    print('Deleting %d processed raw frames...' % len(raw_frames_files))
    os.system('bash delete_processed_raw_frames.sh')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Resize extracted raw frames online')
    parser.add_argument(
        '--raw-frames-path',
        default='/home/wangxiao13/dataset/download_video/video_rawframes',
        help='path to MMU extraced frames')
    parser.add_argument(
        '--resized-frames-root',
        default='/home/wangxiao13/annotation/data/relevance/raw_frames',
        help='path to store resized frames')
    parser.add_argument(
        '--short-side', type=int, default=256, help='short side of video')
    parser.add_argument(
        '--num-workers', type=int, default=8, help='number gpus')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    short_side = args.short_side
    num_workers = args.num_workers
    raw_frames_path = args.raw_frames_path
    resized_frames_root = args.resized_frames_root

    epoch = 1
    while True:
        print(f'Epoch {epoch} ... ')
        process_raw_frames(short_side, num_workers, raw_frames_path,
                           resized_frames_root)
        epoch += 1
        time.sleep(60)
