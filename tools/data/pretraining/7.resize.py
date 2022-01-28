import os
import sys
sys.path.append('/home/wangxiao13/annotation/mmaction2/tools/data/pretraining/lumo/lumo')
import time
from contextlib import contextmanager

import PIL.Image
from torchvision.transforms import Resize
from lumo import Params, Logger
from lumo.utils.filebranch import FileBranch
from joblib import hash, Parallel, delayed

log = Logger()

# pip install accelerate
class FB(FileBranch):

    def branch(self, *name):
        res = FileBranch(os.path.join(self.root, *name), touch=False, listener=self._listeners)
        self.send(res.root)
        return res


@contextmanager
def cached(fn):
    import shutil
    cache_fn = f'{fn}.lumo_cache'
    try:
        yield cache_fn
    except Exception as e:
        os.remove(cache_fn)
        raise e
    finally:
        if os.path.exists(cache_fn):
            shutil.move(cache_fn, fn)


def apply(f, tgt_file, i):
    log.info(i, f, tgt_file)
    try:
        img = PIL.Image.open(f)  # type: PIL.Image.Image
    except (FileNotFoundError, PIL.UnidentifiedImageError, IsADirectoryError) as e:
        log.info(i, e, f)
        return
    except OSError as e:
        log.info(i, e, f)
        return

    resize = Resize(256)

    os.makedirs(os.path.dirname(tgt_file), exist_ok=True)
    try:
        with cached(tgt_file) as tgt_file:
            resize(img).save(tgt_file, format='JPEG')
        os.remove(f)
    except Exception as e:
        log.info(i, e, f)
        return


def main():
    pm = Params()
    pm.index = 0
    pm.workers = 1
    pm.resize = 256
    pm.num_workers = 12
    pm.root = '/home/wangxiao13/dataset/download_video/unresized_frames_hetu'
    pm.tgt_root = '/home/wangxiao13/annotation/data/hetu_human/frames'
        
    pm.from_args()
    log.raw(pm)
    resize = Resize(pm.resize)

    fb = FB(pm.root)

    def create_iter():
        for i, f in enumerate(fb.find_file_in_depth('.jpg', 1)):
            # sample f /home/wangxiao13/dataset/download_video/video_rawframes/8/60986300568_210.jpg
            if int(hash(f)[-6:-4], 16) % pm.workers != pm.index:
                continue
            tgt_file = f.replace(pm.root, pm.tgt_root)
            yield delayed(apply)(f, tgt_file, i)

    while True:
        Parallel(pm.num_workers, verbose=3)(create_iter())
        time.sleep(120)
        log.info('reloop')


if __name__ == '__main__':
    main()