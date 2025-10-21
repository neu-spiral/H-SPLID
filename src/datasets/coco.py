# src/datasets/coco.py
import torch
import torchvision
import os
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union
import numpy as np
from pycocotools.coco import COCO
from src.datasets.non_unique_coco_image_ids import NON_UNIQUE_COCO_ANIMALS_IDS

import fiftyone.zoo as foz

import fiftyone as fo

from PIL import Image
from fiftyone import ViewField

def _get_merged_mask(coco, annotations):
    mask = None
    # aggregate all masks
    for i in range(len(annotations)):
        mask_i = coco.annToMask(annotations[i]) > 0
        if mask is None:
            mask = mask_i
        else:
            mask += mask_i
    # convert to int tensor
    mask_tensor = torch.from_numpy(mask).unsqueeze(0).int()
    return mask_tensor

def _extract_id_from_path(path):
    return int(path.split("/")[-1].split(".")[0].lstrip("0"))

def get_path_by_image_id(paths, id):
    # adds leading zeros to get full id of length 12
    padded_id = id.zfill(12)
    print(padded_id)
    matches = []
    for path_i in paths:
        if padded_id in path_i:
            matches.append(path_i)
    if len(matches) > 1:
        raise ValueError(f"Image with {id} is not unique found {len(matches)} matches")
    else:
        return matches[0]

def remove_non_uniquely_labeled_images(image_paths):
    for id in NON_UNIQUE_COCO_ANIMALS_IDS:
        for i, path_i in enumerate(image_paths):
            if str(id).zfill(12) in path_i:
                del image_paths[i]
    return image_paths

class CocoAnimals(torch.utils.data.Dataset):
    """`MS Coco Detection <https://cocodataset.org/#detection-2016>`_ Dataset.

    It requires the `COCO API to be installed <https://github.com/pdollar/coco/tree/master/PythonAPI>`_.

    Args:
        root (str or ``pathlib.Path``): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
        self,
        root: Union[str, Path],
        annFile: str = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        train: bool = True,
    ) -> None:
        # super().__init__(root, None, transform, target_transform)

        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        
        self.classes = ["bear", "elephant", "giraffe", "zebra"]
        
        self.root = Path(root)
        self.dir = self.root / "coco-2017"
        
        # overvrite foz download path
        fo.config.dataset_zoo_dir = self.root
        # check here if download is really necessary to save some time as foz download check takes about 10s
        if download and self.dir.exists():
            download = False
        self.full_dataset = foz.load_zoo_dataset(
                "coco-2017",
                split="train",
                label_types=["detections"],
                classes=["zebra", "giraffe", "elephant", "bear"],
                download_if_necessary=download,
            )
            
        self.filtered_data = self.full_dataset.filter_labels("ground_truth", ViewField("label").is_in(self.classes))
        self.image_paths = self.filtered_data.values("filepath")
        self.train_split_ratio = 0.8
        self.image_paths = self.create_split(self.train, self.train_split_ratio)
        self.image_paths = remove_non_uniquely_labeled_images(self.image_paths)
        
        if annFile is None:
            self.annFile = self.dir / 'raw/instances_train2017.json'
        else:
            self.annFile = annFile
        
        self.coco = COCO(self.annFile)
        
        # bear - label: 0, category: 23 
        # elephant - label: 1, category: 22 
        # giraffe - label: 2, category: 25
        # zebra - label: 3, category: 24
        self.coco_categories = [23, 22, 25, 24]
        # self._semantic_seg_classes_map = {c: i for i, c in enumerate(["background"] + self.classes)}
        self.class_map = {c: i for i, c in enumerate(self.classes)}
        self.coco_label_to_categories_map = {self.class_map[c]: cc for c, cc in zip(self.classes, self.coco_categories)}
        self.coco_categories_to_labels_map = {value: key for key, value in self.coco_label_to_categories_map.items()}

        # used if no resize is specified
        self.default_resize = torchvision.transforms.Resize((224, 224))
        # just for debugging and checking if labels are unique per image
        self._non_unique_images_ids = []
    
    def create_split(self, train: bool, train_ratio: float):
        # always use same rng to get the same split
        rng = np.random.RandomState(1)
        n_samples = len(self.image_paths)
        split_ratio = train_ratio if train else (1 - train_ratio)
        random_indices = rng.randint(0, n_samples, size=int(split_ratio*n_samples))
        return [self.image_paths[i] for i in random_indices]
    
    def load_by_image_id(self, image_id: str):
        path = get_path_by_image_id(self.image_paths, image_id)
        return self._load_image(path)
    
    def _load_image(self, path) -> Image.Image:
        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def _load_label_and_annotation(self, image_id: int) -> List[Any]:
        label_to_annotations = {}
        labels = []
        for category_i in self.coco_categories:
            anns_ids = self.coco.getAnnIds(imgIds=image_id, catIds=category_i, iscrowd=None)
            # check if list is empty or contains annotation
            if anns_ids:
                annotations = self.coco.loadAnns(anns_ids)
                label_candidates = []
                for ann in annotations:
                    label_candidates.append(self.coco_categories_to_labels_map[ann["category_id"]])
                # find the most common label and define it as image label
                labels.append(torch.mode(torch.as_tensor(label_candidates), 0).values.item())
                label_to_annotations[labels[-1]] = annotations

        
        # Again, find the most common label and define it as image label
        label = torch.mode(torch.as_tensor(labels), 0).values.item()
        annotation = label_to_annotations[label]
        if len(label_to_annotations) > 1 or len(labels) > 1:
            self._non_unique_images_ids.append(image_id)
            # for debugging and collecting all non_unique images remove the ValueError
            raise ValueError(f"Labeling is not unique for image: {image_id} with labels: {str(labels)} and majority label {label}")
        return torch.tensor(label), annotation
        

    def _load_mask(self, annotations) -> List[Any]:
        mask = _get_merged_mask(self.coco, annotations)
        return mask   

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:

        if not isinstance(index, int):
            raise ValueError(f"Index must be of type integer, got {type(index)} instead.")
        path = self.image_paths[index]
        image_id = _extract_id_from_path(path)
        image = self._load_image(path)
        label, annotation = self._load_label_and_annotation(image_id)
        mask = self._load_mask(annotation)
        # need to transform to pil image for resizing and other transformations to work
        mask = torchvision.transforms.functional.to_pil_image(mask)
        
        if self.transform is not None:
            image = self.transform(image)
        else:
            # images need to be always in the same shape
            image = self.default_resize(image)
        
        if self.target_transform is not None:
            mask = self.transform(mask)
        else:
            # mask needs to be always in the same shape as image
            mask = self.default_resize(mask)
        return image, label, mask

    def __len__(self) -> int:
        return len(self.image_paths)
