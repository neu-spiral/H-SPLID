from pathlib import Path
import torch
import torch.utils.data as data

from torchvision.datasets.folder import default_loader
from torchvision import transforms as tf
from typing import Callable, Optional
import numpy as np

class ImageNet(data.Dataset):
    def __init__(
        self, 
        root: str = './assets/datasets',
        subset_path: Optional[str] = None,  # If provided, only load specified classes
        train: bool = True,
        transform: Optional[Callable] = None,
        samples_per_class: Optional[int] = None,          # fixed number per class
        sample_fraction_per_class: Optional[float] = None, # fraction [0,1] per class
        sample_fraction: Optional[float] = None,           # fraction [0,1] of total dataset
    ) -> None:
        super().__init__()
        split = 'train' if train else 'val'
        self.root = Path(root) / 'imagenet' / f'ILSVRC2012_img_{split}'
        self.transform = transform
        self.train = train
        self.split = split

        # RNG for reproducibility
        self.rng = np.random.RandomState(1)

        # Determine class subdirectories
        if subset_path is not None:
            with open(subset_path, 'r') as f:
                lines = [l.strip() for l in f if l.strip()]
            subdirs, class_names = zip(*(line.split(' ', 1) for line in lines))
        else:
            subdirs = sorted([p.name for p in self.root.iterdir() if p.is_dir()])
            class_names = subdirs

        # Collect (path, class_idx) tuples
        all_imgs = []
        for idx, subdir in enumerate(subdirs):
            files = sorted((self.root / subdir).glob('*.JPEG'))
            # per-class fixed count
            if samples_per_class is not None:
                files = files[:samples_per_class]
            # per-class fraction
            if sample_fraction_per_class is not None:
                k = int(len(files) * sample_fraction_per_class)
                # sample without replacement
                chosen = self.rng.choice(len(files), size=k, replace=False)
                files = [files[i] for i in chosen]
            # add to list
            all_imgs.extend([(str(fp), idx) for fp in files])

        # global fraction sampling
        if sample_fraction is not None:
            total = len(all_imgs)
            k = int(total * sample_fraction)
            chosen = self.rng.choice(total, size=k, replace=False)
            all_imgs = [all_imgs[i] for i in chosen]

        self.imgs = all_imgs
        self.classes = list(class_names)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index, include_meta: bool = False):
        path, target = self.imgs[index]
        img = default_loader(path)
        im_size = img.size
        class_name = self.classes[target]

        if self.transform is not None:
            img = self.transform(img)
        
        target = torch.tensor(target)
        if include_meta:
            return img, target, {
                'im_size': im_size,
                'index': index,
                'class_name': class_name
            }
        return img, target

    def create_split(self, train: bool, train_ratio: float):
        n = len(self.imgs)
        idxs = self.rng.permutation(n)
        cut = int(n * train_ratio)
        chosen = idxs[:cut] if train else idxs[cut:]
        return [self.imgs[i] for i in chosen]
