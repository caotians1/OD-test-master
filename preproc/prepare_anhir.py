import os
import os.path as osp
import csv
from PIL import Image
import argparse
from tqdm import tqdm
import pickle

import numpy as np
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir', default="E:\\ANHIR\\")
    parser.add_argument('--proc_path', default="E:\\ANHIR\\images_96.npy")
    args = parser.parse_args()
    source_dir = args.source_dir

    if os.name == "posix":
        split_char = "/"
    else:
        split_char = "\\"
    A = []
    labels = []
    dir_level = len(source_dir.split(split_char))
    subdirs = ["breast_ER_patches", "breast_HE_patches", "kidney_HE_patches", "kidney_MAS_patches"]

    for subdir in subdirs:

        for (dirpath, dirnames, filenames) in os.walk(osp.join(source_dir, subdir)):
            for filename in filenames:
                if '.jpg' in filename:
                    dir_strs = dirpath.split(split_char)[dir_level:]
                    labels.append(subdir)
                    with open(osp.join(dirpath, filename), 'rb') as f:
                        with Image.open(f) as img:
                            #img = img.resize((224, 224))
                            #fn = filename.replace(" ", "_")
                            A.append(np.array(img))

    np.save(args.proc_path, np.stack(A))
    with open(osp.join(source_dir, "labels.pkl"),  "wb") as fp:
        pickle.dump(labels, fp)


