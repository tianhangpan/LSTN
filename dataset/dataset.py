import torch
from torchvision import transforms
import cv2
import numpy as np
from torch.utils.data import Dataset
from itertools import chain

from dataset.image import LoadData


class BaseDataset(Dataset):
    def __init__(self, root, mask, train):
        super().__init__()
        self.lines = root
        self.mask = mask
        self.train = train
        self.front_boundaries = []
        self.back_boundaries = []
        last_back_boundary = -1
        self.len_lines = [len(line_ele) for line_ele in self.lines]
        for len_line_ele in self.len_lines:
            self.front_boundaries.append(last_back_boundary + 1)
            self.back_boundaries.append(last_back_boundary + len_line_ele)
            last_back_boundary = self.back_boundaries[-1]
        self.lines = list(chain(*self.lines))
        self.rand_mask = list(range(len(self.lines)))

    def __len__(self):
        return len(self.rand_mask)

    def __getitem__(self, index):
        pass

    @staticmethod
    def trans(img, target, mask):
        target = cv2.resize(target, (target.shape[1] // 16, target.shape[0] // 16)) * (16 * 16)

        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                    std=[0.229, 0.224, 0.225])])
        img = np.array(img)
        img = transform(img)
        target = torch.tensor(target).type(torch.FloatTensor).unsqueeze(0)
        if not isinstance(mask, type(None)):
            mask = cv2.resize(mask, (mask.shape[1] // 16, mask.shape[0] // 16))
            mask = torch.tensor(mask).unsqueeze(0)
        else:
            mask = torch.ones_like(target)

        return img, target, mask


class PresentDataset(BaseDataset):
    def __init__(self, root, mask, train):
        super().__init__(root, mask, train)
        self.rand_mask.sort()
        if self.train:
            if len(self.rand_mask) < 3000:
                self.rand_mask *= 4

    def __getitem__(self, index):
        img, target, mask = LoadData.test_data(self.lines[self.rand_mask[index]], self.mask)
        return self.trans(img, target, mask)


class PresentNextDataset(BaseDataset):
    def __init__(self, root, mask, train):
        super().__init__(root, mask, train)
        self.rand_mask = list(set(self.rand_mask) - set(self.back_boundaries))
        self.rand_mask.sort()
        if self.train:
            if len(self.rand_mask) < 3000:
                self.rand_mask *= 4

    def __getitem__(self, index):
        raw_img, target, mask = LoadData.test_data(self.lines[self.rand_mask[index]], self.mask)
        _, next_target, _ = LoadData.test_data(self.lines[self.rand_mask[index]+1], None)
        img, target, mask = self.trans(raw_img, target, mask)
        _, next_target, _ = self.trans(raw_img, next_target, None)
        return img, target, next_target, mask
