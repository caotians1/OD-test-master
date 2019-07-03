import os
import os.path as osp
import csv
from PIL import Image
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir', default="E:\\NIHCC")
    parser.add_argument('--index_file', default="Data_Entry_2017.csv")
    parser.add_argument('--image_dir', default="images")
    parser.add_argument('--proc_dir', default="E:\\NIHCC\\images_224")
    args = parser.parse_args()
    source_dir = args.source_dir
    index_file = args.index_file
    image_dir = args.image_dir
    processed_image_dir = args.proc_dir
    if not osp.exists(processed_image_dir):
        os.mkdir(processed_image_dir)
    img_list = []
    with open(osp.join(source_dir, index_file), 'r') as fp:
        csvf = csv.DictReader(fp)
        for row in csvf:
            imp = osp.join(source_dir, image_dir, row['Image Index'])
            if osp.exists(imp):
                img_list.append(row['Image Index'])

    for im in img_list:
        imp = osp.join(source_dir, image_dir, im)
        with open(imp, 'rb') as f:
            with Image.open(f) as img:
                img = img.convert('L').resize((224,224))
        img.save(osp.join(source_dir, processed_image_dir, im))
