import torch
from torchvision import transforms, datasets
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np

class CMNIST(datasets.MNIST):
    def __init__(
        self,
        root: Union[str, Path],
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root=root, transform=transform, target_transform=target_transform, train=train, download=download)

        # NOTE: This is run in datasets.MNIST superclass
        # if self._check_legacy_exist():
        #     self.data, self.targets = self._load_legacy_data()
        #     return

        # if download:
        #     self.download()

        # if not self._check_exists():
        #     raise RuntimeError("Dataset not found. You can use download=True to download it")

        # self.data, self.targets = self._load_data()
        
        # construct Concatenated MNIST by overwriting self.data and self.label from super class
        self.data, self.targets = _create_random_pairings(self.data, self.targets, random_state=1)
        
        # use left labels only
        self.targets = self.targets[:, 0]
        



def _add_padding(data, pad_to, pad_value):
    """Assumes data shape of -1 x C x W x H and pad_to=(W_new, H_new)"""
    assert len(data.shape) == 4
    diffs = []    
    for i in [1,0]:
        if data.shape[2+i] < pad_to[i]:
            diff = pad_to[i] - data.shape[2+i] 
            if diff % 2 == 0:
                diffs += [diff // 2, diff // 2]
            else:
                diffs += [diff // 2, diff // 2 + 1]
        else:
            diffs += [0,0]
    # pad last dim by last two entries in the list and 2nd to last by 2nd to last to entries and so on
    # e.g.: for diffs=[1, 1, 2, 2] pad last dim by [1, 1] and 2nd to last by [2, 2]
    data = torch.nn.functional.pad(data, diffs, "constant", pad_value)
    return data

def _create_random_pairings(data, labels, random_state=None, data_cat_dim=3, pad_to=(64,64)):
    """Creates a random pairings between two images
       Assumes data shape of -1 x C x W x H and pad_to=(W_new, H_new)
    """
    if random_state is None:
        random_state = np.random.randint(100000)
    if len(data.shape) == 3:
        # add color dimension as 1
        data = data.unsqueeze(1)
    rng = np.random.default_rng(random_state)
    left_idx = rng.choice(data.shape[0], data.shape[0], replace=False)
    right_idx = rng.choice(data.shape[0], data.shape[0], replace=False)

    left_data = data[left_idx].clone()
    right_data = data[right_idx].clone()
    left_labels = labels[left_idx].clone()
    right_labels = labels[right_idx].clone()
    concat_data = torch.cat([left_data, right_data], data_cat_dim)
    concat_labels = torch.stack([left_labels, right_labels], dim=1)
    if pad_to is not None:
        # adds black padding to make images squares
        concat_data = _add_padding(concat_data, pad_to, pad_value=0)
    # remove additional dimesion again
    concat_data = concat_data.squeeze(1)
    return concat_data, concat_labels