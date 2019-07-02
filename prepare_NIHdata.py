import os
import os.path as osp
import csv
from PIL import Image

source_dir = "E:\\NIHCC"
index_file = "Data_Entry_2017.csv"
image_dir = 'images'
processed_image_dir = osp.join(source_dir, 'images_224')
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
