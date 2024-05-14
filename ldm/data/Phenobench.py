import os
from re import L
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import random

# Some words may differ from the class names defined in COCO-Stuff to minimize ambiguity
Pheno_dict = {
'0':'background',
'1':'crop',
'2':'weed',
'3':'partial-crop',
'4':'partial-weed',
}


class Phenobench(Dataset):
    def __init__(self,
                 data_root,
                 txt_file,
                 size=None,
                 interpolation="bicubic",
                 flip_p=0.5
                 ):
        self.data_root = data_root
        self.data_paths = txt_file
        with open(self.data_paths, "r") as f:
            self.image_paths = f.read().splitlines()
        self._length = len(self.image_paths)
        self.size = size
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        self.flip_p = flip_p

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict()
        path = self.image_paths[i]
        pil_image = Image.open(path)
        if not pil_image.mode == "RGB":
            pil_image = pil_image.convert("RGB")

        path_ = self.image_paths[i][:-4]
        if 'train' in path_:
            path2 = os.path.join(self.data_root, 'annotations', 'training', path_.split('/')[-1] + '.png')
        else:
            path2 = os.path.join(self.data_root, 'annotations', 'validation', path_.split('/')[-1] + '.png')
        pil_image2 = Image.open(path2)

        flip = random.random() < self.flip_p
        if self.size is not None:
            pil_image = pil_image.resize((self.size, self.size), resample=self.interpolation)
            pil_image2 = pil_image2.resize((self.size, self.size), resample=PIL.Image.NEAREST)
        if flip:
            pil_image = pil_image.transpose(PIL.Image.FLIP_LEFT_RIGHT)
            pil_image2 = pil_image2.transpose(PIL.Image.FLIP_LEFT_RIGHT)
        image = np.array(pil_image).astype(np.uint8)
        example["image"] = (image / 127.5 - 1.0).astype(np.float32)
        label = np.array(pil_image2).astype(np.float32)
        example["label"] = label
        class_ids = sorted(np.unique(label.astype(np.uint8)))
        # print(class_ids, path)
        if class_ids[-1] == 255:
            class_ids = class_ids[:-1]
        # if 128 in class_ids:
        #     class_ids = np.delete(class_ids, np.where(class_ids == 128))
        class_ids_final = np.zeros(5)
        text = ''
        for i in range(len(class_ids)):
            text += Pheno_dict[str(class_ids[i])]
            text += ' '
            class_ids_final[class_ids[i]] = 1
        # text = text[:-1]
        name = os.path.basename(path)
        if '05-15_' in name:
            text = text + 'sunny morning'
        elif '05-26_' in name:
            text = text + 'sunny noon'
        elif '06-05_' in name:
            text = text + 'cloudy wet'
        else:
            text = text[:-1]
        class_ids_final = np.concatenate((class_ids_final, [-1]), axis=0)
        example["caption"] = text
        example["class_ids"] = class_ids_final

        return example


class PhenoTrain(Phenobench):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class PhenoVal(Phenobench):
    def __init__(self, flip_p=0., **kwargs):
        super().__init__(flip_p=flip_p, **kwargs)
