import cv2
import numpy as np
from os.path import join, basename
from glob import glob
data_root = '/mnt/d/datasets/freestyle_crops'
semantics = join(data_root, 'annotations', 'training')
label_files = glob(join(semantics, '*.png'))

# phenobench semantics 
# background 0
# crop 1
# weed 2
# partial crop 3
# partial weed 4
# crop 1+3
# weed 2+4
# for file in label_files:
#     img = cv2.imread(file, cv2.IMREAD_UNCHANGED)    # Note! 16-bit single channel image. 
#     print(np.unique(img[:,:,0]), np.unique(img[:,:,1]), np.unique(img[:,:,2]))
    # cv2.imshow('label_mask', img)
    # cv2.waitKey(0)

label_coco = glob('/mnt/d/datasets/cocostuff/val/label/*.png')
for file in label_coco:
    img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
    cv2.imshow('label_mask', img)
    cv2.waitKey(0)