import os
import glob
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from utils import labelMap, invLabelMap, readAnotationTxt


class CardsDataset(object):
    def __init__(self, img_dir, anotation_dir, labels_path, transform=None):
        self.img_dir = img_dir
        self.anotation_dir = anotation_dir
        if transform == None:
            # Normalization values
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            self.transform = transforms.Compose([
                transforms.Resize((300, 300)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
                ])
        else:
            self.transform = transform

        # load all image files, sorting them to ensure that they are aligned
        self.imgs = list(sorted(glob.glob(os.path.join(self.img_dir, '*.jpeg'))))
        self.annotations = list(sorted(glob.glob(os.path.join(self.anotation_dir, '*.txt'))))
        self._labels = labelMap(os.path.join(labels_path))
        self._inv_labels = invLabelMap(self._labels)

        assert len(self.imgs) == len(self.annotations)

    def __getitem__(self, idx):
        # select image and anotation index
        img_path = os.path.join(self.imgs[idx])
        anotation_path = os.path.join(self.annotations[idx])
        # load image in RGB and anotation
        img = Image.open(img_path).convert("RGB")
        labels, boxes = readAnotationTxt(anotation_path)
       
        # convert boxes and labels into a torch.Tensor
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int32)

        img = self.transform(img)

        return img, boxes, labels

    def __len__(self):
        return len(self.imgs)

    @property
    def labels(self):
        return self._labels

    @property
    def invLabels(self):
        return self._inv_labels
    
    def fileName(self, idx):
        return self.imgs[idx]



    




