from torch.utils.data import Dataset
import random
from PIL import Image
import numpy as np
import h5py
import cv2
import os

class listDataset(Dataset):
    def __init__(self, root, shape = None, shuffle = True, transform = None,
                 train = False, seen = 0, batch_size = 1, num_workers = 4):
        if train:
            root = root * 4
        random.shuffle(root)
        self.nSamples = len(root)
        self.lines = root
        self.transform = transform
        self.train = train
        self.shape = shape
        self.seen = seen                    # 0?
        self.batch_size = batch_size
        self.num_workers = num_workers      # 4?

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        img_path = self.lines[index]
        gt_path = img_path.replace('images', 'ground_truth')
        gt_path = os.path.splitext(gt_path)[0] + '.h5'
        img = Image.open(img_path).convert('RGB')       # RGB mode, (w,h)
        gt_file = h5py.File(gt_path)
        target = np.asarray(gt_file['density'])                     # (h, w)
        ## pooling effect
        shape1 = int(target.shape[1] / 8.0)                         # w
        shape0 = int(target.shape[0] / 8.0)                         # h
        target = cv2.resize(target, (shape1, shape0)) * 64    # (h/8, w/8)
        if self.transform is not None:
            img = self.transform(img)               # torch.Size([3,h,w])
        return img,target
