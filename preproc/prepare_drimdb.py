import os
import os.path as osp
import csv
from PIL import Image
import argparse
from tqdm import tqdm
import numpy as np
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir')
    parser.add_argument('--proc_path', default="E:\\DRIMDB\\images_224.npy")
    args = parser.parse_args()
    source_dir = args.source_dir

    if os.name == "posix":
        split_char = "/"
    else:
        split_char = "\\"
    A = []
    dir_level = len(source_dir.split(split_char))
    for (dirpath, dirnames, filenames) in os.walk(source_dir):
        for filename in filenames:
            if '.jpg' in filename:
                dir_strs = dirpath.split(split_char)[dir_level:]
                if "Good" in dir_strs:
                    continue
                with open(osp.join(dirpath, filename), 'rb') as f:
                    with Image.open(f) as img:
                        img = img.resize((224, 224))
                        fn = filename.replace(" ", "_")
                        A.append(np.array(img))

    np.save(args.proc_path, np.stack(A))



