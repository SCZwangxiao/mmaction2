import io
import os
import time
import glob
from tqdm import tqdm
import multiprocessing as mp

import decord

video_file_path = "/home/wangxiao13/annotation/mmaction2/data/querytag/videos"

video_files = glob.glob(os.path.join(video_file_path, "*.mp4"))
print(f'Checking video files in {video_file_path}...')


def check_videos(video_file):
    try:
        with open(video_file, 'rb') as F:
            container = decord.VideoReader(F, num_threads=2)
    except:
        print(f"Error in : {video_file}")
        os.system(f"rm -rf {video_file}")


pool = mp.Pool(mp.cpu_count()>>1)

for video_file in video_files:
    pool.apply_async(func=check_videos, args=(video_file, ))


prev_count = len(video_files)
pbar = tqdm(total=len(video_files))
while True:
    time.sleep(1)
    process_count = len(pool._cache)
    pbar.update(prev_count - process_count)
    prev_count = process_count
    if process_count == 0:
        print("all processes are finished")
        pbar.close()
        break

pool.close()
pool.join()