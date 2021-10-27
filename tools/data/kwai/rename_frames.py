import argparse
import glob
import os
import os.path as osp
import re
import time

from tqdm import tqdm


def process_resized_frames(raw_frames_path, num_clips):
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
                old_filename = filename_tpl.format(pid, pid, frame_id)
                new_filename = filename_tpl.format(pid, pid, i + 1)
                rename_commands.append(f'mv {old_filename} {new_filename}')

    with open('rename_resized_frames.sh', 'w') as F:
        for cmd in rename_commands:
            F.write(f'{cmd}\n')
    print('Renaming %d resized raw frames...' % len(rename_commands))
    os.system('bash rename_resized_frames.sh')

    with open('pid_with_missing_frames', 'w') as F:
        for pid in pid_missing_frames:
            F.write(f'{pid}\n')

    with open('pid_renamed', 'w') as F:
        for pid in pid_renamed:
            F.write(f'{pid}\n')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Rename resized raw frames online')
    parser.add_argument(
        '--resized-frames-root',
        default='/home/wangxiao13/annotation/data/relevance/raw_frames',
        help='path of resized frames')
    parser.add_argument(
        '--num-clips', type=int, default=8, help='path of resized frames')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    resized_frames_root = args.resized_frames_root
    num_clips = args.num_clips

    epoch = 1
    while True:
        print(f'Epoch {epoch} ... ')
        process_resized_frames(resized_frames_root, num_clips)
        epoch += 1
        time.sleep(60)
