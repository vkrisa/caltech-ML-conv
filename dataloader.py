import pathlib
import pandas as pd
import numpy as np
import torch
import torchvision

from PIL import Image
from torch.utils.data import Dataset


class CaltechDataset(Dataset):
    def __init__(self, root):
        self._width = 200
        self._height = 200
        self._root = pathlib.Path(root)
        self._files = tuple(self._traverse(self._root))
        self._data = self._transform(self._files)
        self.augment = True
        self.trans = torchvision.transforms.RandomApply([
            torchvision.transforms.RandomAffine(15),
            torchvision.transforms.RandomPerspective(p=0.5),
            torchvision.transforms.RandomRotation([-15, 15], resample=Image.BICUBIC),
            torchvision.transforms.RandomResizedCrop(size=200, scale=(0.6, 8)),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomGrayscale()

        ])

    def __len__(self):
        return len(self._files)

    def __getitem__(self, item):
        label, image = self._data[item]
        image = self._image_to_array(image)
        return torch.tensor(image, dtype=torch.float), self._label_tensor(label)

    def _transform(self, files):
        labels = tuple(map(lambda x: x.split('/')[1].split('.')[0], files))
        df = pd.DataFrame({"label": labels, "path": files})
        np_array = df.to_numpy()
        return np_array

    def _traverse(self, root: pathlib.Path):
        for entry in root.iterdir():
            if entry.is_file() and entry.suffix == '.jpg':
                yield str(entry)
            elif entry.is_dir():
                yield from self._traverse(entry)

    def _label_tensor(self, label):
        return torch.tensor(int(label) - 1).long()

    def _image_to_array(self, item):
        image = Image.open(item).convert('RGB')
        image = image.resize((self._width, self._height), Image.ANTIALIAS)
        if self.augment:
            image = self.trans(image)
        image = torchvision.transforms.ToTensor()(image)

        return image
