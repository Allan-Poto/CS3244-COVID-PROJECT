from torch.utils.data import Dataset
from torchvision import transforms
import torchvision
import os
import matplotlib.pyplot as plt
import torch
import cv2
from . import FILE_COL, LABEL_COL


"""
ds1 = XrayDataset('.', pd.DataFrame())
len(ds1) ==> __len__(self)
ds1[idx] ==> __getitem__(self, idx)
"""

CLASS_TO_IDX = {'covid': 0, 'normal': 1, 'viral_pneumonia': 2}


IMAGENET_NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])

BIG_SIZE = 224
SIZE = 224


def up_contrast(i):
    return transforms.functional.adjust_contrast(i, 1.5)


DEFAULT_TRANSFORMS = transforms.Compose([
    transforms.ToPILImage(),
    # transforms.Grayscale(3),
    transforms.Resize((224, 224)),
    # transforms.Lambda(up_contrast),
    transforms.ToTensor(),
    IMAGENET_NORMALIZE
])

tfms = transforms.Compose([
    transforms.ToPILImage(),
    # transforms.Grayscale(3),
    transforms.Resize((240, 240)),
    transforms.RandomCrop((224, 224)),
    # transforms.RandomHorizontalFlip(0.1),
    transforms.RandomRotation(5),
    # transforms.Lambda(up_contrast),
    transforms.ColorJitter(0.01, 0.01, 0, 0),
    transforms.ToTensor(),
    IMAGENET_NORMALIZE
])


class XrayDataset(Dataset):
    def __init__(self, root, file_col, label_col, df, tfms=tfms, train=False):
        self.root = root  # DATA_DIRECTORY
        self.df = df  # contains all your filenames and labels
        self.filepaths = df[file_col].values
        uniq_labels = sorted(list(set(df[label_col].values)))
        self.class_names = uniq_labels
        self.class_idx = dict([(e, i) for i, e in enumerate(uniq_labels)])
        self.labels = [torch.tensor(self.class_idx[l], dtype=torch.long)
                       for l in df[label_col].values]
        self.train = train
        self.aug = tfms

    def show_img(self, idx):
        fp = self.filepaths[idx]
        img = cv2.imread(self.root + fp)
        print(plt.imshow(img))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        fp = self.filepaths[idx]
        label = self.labels[idx]  # integer
        img = cv2.imread(os.path.join(self.root, fp))  # numpy array
        # we can make it smaller for faster processing
        # shape -> [h, w, channel]
        # 1. change to tensor
        # 2. change the shape
        img = self.aug(img) if self.train else DEFAULT_TRANSFORMS(img)
        # tensor expected shape -> [channel, h, w]
        return img, label
