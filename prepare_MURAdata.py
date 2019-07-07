import os
import os.path as osp
import csv
from PIL import Image
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir', default="E:\\MURA-v1.1")
    parser.add_argument('--index_file', default="train_image_paths.csv")
    parser.add_argument('--image_dir', default="train")
    parser.add_argument('--proc_dir', default="E:\\MURA-v1.1\\images_224")
    args = parser.parse_args()
    source_dir = args.source_dir
    index_file = args.index_file
    image_dir = args.image_dir
    processed_image_dir = args.proc_dir

    img_list = []
    with open(osp.join(source_dir, index_file), 'r') as fp:
        csvf = csv.DictReader(fp, ['Image Path', ])
        for row in csvf:
            imp = osp.join(source_dir, row['Image Path'][10:])
            if osp.exists(imp):
                img_list.append('/'.join(row['Image Path'][10:].split('/')[1:]))

    for im in img_list:
        imp = osp.join(source_dir, image_dir, im)
        with open(imp, 'rb') as f:
            with Image.open(f) as img:
                img = img.convert('L').resize((224,224))
        if not osp.exists(osp.dirname(osp.join(source_dir, processed_image_dir, im))):
            os.makedirs(osp.dirname(osp.join(source_dir, processed_image_dir, im)), exist_ok=True)
        img.save(osp.join(source_dir, processed_image_dir, im))
