from glob import glob
import PIL
from PIL import Image
import os
import numpy as np


labels = glob('/netscratch/naeem/cocostuff/phenobench_cocostuff/annotations/validation/*.png')
for file in labels:
    pil_image = Image.open(file)
    label = np.array(pil_image).astype(np.float32)
    class_ids = sorted(np.unique(label.astype(np.uint8)))
    if 128 in class_ids:
        print(file)
        print('hold')
