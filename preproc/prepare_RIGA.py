import os
import os.path as osp
import csv
from PIL import Image
import argparse
from os import walk
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir')
    #parser.add_argument('--index_file', default="train_image_paths.csv")
    parser.add_argument('--image_dir', default="train")
    parser.add_argument('--proc_dir', default="E:\\MURA-v1.1\\images_224")
    args = parser.parse_args()
    source_dir = args.source_dir
    #index_file = args.index_file
    image_dir = args.image_dir
    processed_image_dir = args.proc_dir

    img_list = []
    for (dirpath, dirnames, filenames) in walk(source_dir):
        for name in filenames:
            if 'prime' in name:
                img_list.append(os.path.join(dirpath, name))

    if not osp.exists(osp.join(source_dir, processed_image_dir)):
        os.makedirs(osp.join(source_dir, processed_image_dir), exist_ok=True)

    for i, im in tqdm(enumerate(img_list)):
        imp = im
        with open(imp, 'rb') as f:
            with Image.open(f) as img:
                img = img.resize((224,224))
        img.save(osp.join(source_dir, processed_image_dir, "image_%i.jpg" % i))
