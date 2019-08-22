from PIL import Image
import numpy as np
from PIL import ImageFilter

source_patchsize = 92
output_size = 96
output_mode = "RGB"
targetNperimage = 210


for img_num in range(1,6):
    source_img = "E:\\ANHIR\\kidney_MAS\\MAS_%d.jpg"%img_num
    img1 = Image.open(source_img)
    hc, sc, vs = img1.convert('HSV').split()
    sc = sc.filter(ImageFilter.GaussianBlur(10))
    W, H = img1.size
    patches = []
    while len(patches) < targetNperimage:
        w = np.random.random() * (W - source_patchsize)
        h = np.random.random() * (H - source_patchsize)
        s = sc.crop((w, h, w+source_patchsize, h+source_patchsize))
        s = np.array(s)
        t = np.array(s > 30).sum()
        if t > 0.1*source_patchsize**2:
            patch = img1.crop((w, h, w+source_patchsize, h+source_patchsize)).resize((output_size, output_size))
            patches.append((w, h))
            patch.save("E:\\ANHIR\\kidney_MAS_patches\\img_%d_w%d_h%d.jpg" % (img_num, w, h))



