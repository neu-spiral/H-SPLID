from typing import Literal, Optional

import pytorch_lightning as pl
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms

from src.datasets.cmnist import CMNIST
from src.datasets.coco import CocoAnimals
from src.datasets.counteranimal import CounterAnimal
from src.datasets.imagenet import ImageNet
from src.datasets.imagenet9 import ImageNet9
from src.datasets.isic2017 import ISIC2017

def get_train_val_split(base_train: torch.utils.data.Dataset, split_seed: int, val_split_ratio: float):
    val_size = int(val_split_ratio * len(base_train))
    train_size = len(base_train) - val_size
    _val_split_seed = torch.Generator().manual_seed(split_seed)
    base_train, base_val = torch.utils.data.random_split(base_train, [train_size, val_size], generator=_val_split_seed) 
    return base_train, base_val

def extract_labels_from_dataset(ds):
    if "targets" in ds.dataset.__dict__:
        return ds.dataset.targets
    else:
        labels = []
        for i in range(len(ds)):
            labels.append(ds[i][2])
        return torch.tensor(labels)
    
class KFoldDataModule(pl.LightningDataModule):
    """KFold cross validation in pytorch-lightning
    based on: https://gist.github.com/ashleve/ac511f08c0d29e74566900fd3efbb3ec"""
    def __init__(
            self,
            datamodule: pl.LightningDataModule,
            k: int = 1,  # fold number. NOTE: Needs to be changed in outer loop
            # split needs to be always the same for correct cross validation
            split_seed: int = 1,
            # num_splits = 10 means our dataset will be split to 10 parts
            # so we train on 90% of the data and validate on 10%
            num_splits: int = 10,

        ):
        super().__init__()
        
        self.k = k
        self.split_seed = split_seed
        self.num_splits = num_splits
        assert 1 <= self.k <= self.num_splits, "incorrect fold number"
        
        self.full_dataset = datamodule.train_dataset
        self.test_dataset = datamodule.test_dataset
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage=None):
        if not self.train_dataset and not self.val_dataset:
            # choose fold to train on
            kf = KFold(n_splits=self.num_splits, shuffle=True, random_state=self.split_seed)
            all_splits = [k for k in kf.split(self.full_dataset)]
            train_indexes, val_indexes = all_splits[self.k]
            train_indexes, val_indexes = train_indexes.tolist(), val_indexes.tolist()

            self.train_dataset, self.val_dataset = self.full_dataset[train_indexes], self.full_dataset[val_indexes]

    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=self.pin_memory, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                         num_workers=self.num_workers, pin_memory=True, persistent_workers=True)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False,
                         num_workers=self.num_workers, pin_memory=True, persistent_workers=True)
        


class MultiViewDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, base_transform, aug_transform, target_transform=None):
        self.dataset = dataset
        self.base_transform = base_transform
        self.aug_transform = aug_transform
        self.target_transform = target_transform
        if self.target_transform is None:
            self.target_transform = lambda x: x
    def __getitem__(self, idx):
        # img, label = self.dataset[idx]
        batch = self.dataset[idx]
        if len(batch) == 2:
            img, label = batch
            mask = None
        elif len(batch) == 3: # dataloader: data, target, masks (e.g., coco)
            img, label, mask = batch 
        else:
            raise ValueError(f"Unexpected batch size of {len(batch)} encountered")
        
        if mask is None:
            return self.aug_transform(img), self.base_transform(img), label
        else:
            return self.aug_transform(img), self.base_transform(img), label, self.target_transform(mask)
            

    def __len__(self):
        return len(self.dataset)

class ColorDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_name: Literal["cifar10", "coco-animals", "isic2017"] = "cifar10",
        data_dir: str = "./assets/datasets",
        subset_path: Optional[str] = None, # for imagenet
        sample_fraction_per_class: Optional[float] = None, # for imagenet
        batch_size: int = 256,
        val_split_ratio: float = 0,
        resize: Optional[int] = None,
        num_workers: int = 1,
        channel_means: torch.Tensor = None,
        channel_stds: torch.Tensor = None,
        evaluate_mode: Literal["common","counter","both"] = "both", # for counteranimal
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.subset_path = subset_path
        self.sample_fraction_per_class = sample_fraction_per_class
        self.batch_size = batch_size
        self.resize = resize
        self.num_workers = num_workers
        self.val_split_ratio = val_split_ratio
        # seed fixing the generator
        self.val_split_seed = 1
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.full_dataset = None
        self.test_labels = None
        self.val_labels = None
        # parameters for normalization and denormalization
        self.channel_means = channel_means
        self.channel_stds = channel_stds
        self.normalize_fn = None
        
        self.evaluate_mode = evaluate_mode
        
        self.dataset_map = {
            "cifar10": datasets.CIFAR10,
            "coco-animals": CocoAnimals,
            "imagenet": ImageNet,
            "counteranimal": CounterAnimal,
            "imagenet9": ImageNet9,
            "isic2017": ISIC2017,
        }
    
    def get_normalize_fn(self, data: torch.Tensor):
        self.channel_means = data.mean([0, 2, 3])
        self.channel_stds = data.std([0, 2, 3])
        return transforms.Normalize(self.channel_means, self.channel_stds)
    
    def denormalize_fn(self, tensor: torch.Tensor)->torch.Tensor:
        """This applies an inverse z-transformation to denormalize data
        """
        device = tensor.device
        return tensor.mul(self.channel_stds.to(device)).add(self.channel_means.to(device))
    
    
    def setup(self, stage: str = None):
        dataset_class = self.dataset_map[self.dataset_name]
        
        base_transforms = transforms.Compose([transforms.ToTensor()])
        target_transform = None
        if self.dataset_name in ["coco-animals", "isic2017"]:
            if self.resize is None:
                self.resize = 224
                print("WARNING: Resize was not defined, but is needed for coco, so it was set to the default of 224")
            target_transform_list = [transforms.Resize((self.resize, self.resize)), transforms.ToTensor()]
            target_transform = transforms.Compose(target_transform_list)
            base_transforms = target_transform

        if self.channel_means is None or self.channel_stds is None:
            # NOTE: For large datasets like imagenet it is better to precompute those statistics once to save compute
            if self.dataset_name == "imagenet":
                # Use imagenet normalization from torchvision
                self.normalize_fn = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])    
            else:    
                # create temp loader to compute normalization statistics automatically
                temp_dataset = dataset_class(self.data_dir, train=True, download=True,
                                            transform=base_transforms, target_transform=target_transform)
                
                temp_loader = DataLoader(temp_dataset, batch_size=1000, num_workers=self.num_workers)
                self.normalize_fn = self.get_normalize_fn(next(iter(temp_loader))[0])
        else:
            self.normalize_fn = transforms.Normalize(self.channel_means, self.channel_stds)
            
        if self.dataset_name not in ["imagenet", "counteranimal", "imagenet9"]:
            transform_list = []
            if self.resize:
                transform_list.append(transforms.Resize((self.resize, self.resize)))
            
            transform_list.extend([transforms.ToTensor(), self.normalize_fn])
            base_transform = transforms.Compose(transform_list)
        else:
            transform_list = [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                self.normalize_fn
            ]
            
            base_transform = transforms.Compose(transform_list)
        
        aug_transform = transforms.Compose([
            *transform_list[:-2],
            transforms.RandomResizedCrop((self.resize, self.resize) if self.resize else (32, 32)),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], 
                p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomSolarize(threshold=0.5, p=0.2),
            transforms.ToTensor(),
            self.normalize_fn
        ])

        # Setup
        if self.dataset_name == "imagenet":
            base_test = dataset_class(self.data_dir, subset_path=self.subset_path, train=False, transform=None, sample_fraction_per_class=self.sample_fraction_per_class)
            base_train = dataset_class(self.data_dir, subset_path=self.subset_path, train=True, transform=None, sample_fraction_per_class=self.sample_fraction_per_class)
        elif self.dataset_name == "counteranimal":
            self.test_dataset = dataset_class(self.data_dir, mode=self.evaluate_mode, transform=base_transform)
            return # counteranimal supports only evaluation
        elif self.dataset_name == "imagenet9":
            self.test_dataset = dataset_class(self.data_dir, subset=self.subset_path, split='val', transform=base_transform)
            return # imagenet9 supports only evaluation
        else:
            base_test = dataset_class(self.data_dir, train=False, transform=None)
            base_train = dataset_class(self.data_dir, train=True, transform=None)
        
        if self.val_split_ratio > 0:
            base_train, base_val = get_train_val_split(base_train, self.val_split_seed, self.val_split_ratio)
        else:
            base_val = base_train

        self.train_dataset = MultiViewDataset(base_train, base_transform, aug_transform, target_transform)
        self.val_dataset = MultiViewDataset(base_val, base_transform, base_transform, target_transform)
        self.test_dataset = MultiViewDataset(base_test, base_transform, base_transform, target_transform)
        self.full_dataset = torch.utils.data.ConcatDataset([self.train_dataset, self.test_dataset])
        self.test_labels = extract_labels_from_dataset(self.test_dataset)
        self.val_labels = extract_labels_from_dataset(self.val_dataset)


    def train_dataloader(self):
        dataset = self.train_dataset
        if "debug" in self.dataset_name:
            subsample_indices = torch.randperm(len(dataset))[:self.batch_size*2]
            sampler = SubsetRandomSampler(subsample_indices)
            shuffle = False
        else:
            sampler = None
            shuffle = True
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, sampler=sampler,
                         num_workers=self.num_workers, pin_memory=True, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                         num_workers=self.num_workers, pin_memory=True, persistent_workers=True)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False,
                         num_workers=self.num_workers, pin_memory=True, persistent_workers=True)
    
    def compute_class_weights(self, method='balanced'):
        """
        Compute class weights for handling imbalanced datasets
        
        Args:
            method: 'balanced' for sklearn-style balanced weights, 'inverse' for inverse frequency
        Returns:
            torch.Tensor: class weights
        """
        if self.train_dataset is None:
            raise ValueError("Dataset not setup. Call setup() first.")

        train_labels = []
        for i in range(len(self.train_dataset.dataset)):
            # Handle different batch formats
            batch = self.train_dataset.dataset[i]
            if len(batch) == 2:
                _, label = batch
            elif len(batch) == 3:
                _, label, _ = batch
            else:
                raise ValueError(f"Unexpected batch format: {len(batch)}")
            train_labels.append(label)
        
        train_labels = torch.tensor(train_labels)
        unique_classes = torch.unique(train_labels)
        n_classes = len(unique_classes)
        
        if method == 'balanced':
            class_counts = torch.bincount(train_labels, minlength=n_classes)
            n_samples = len(train_labels)
            weights = n_samples / (n_classes * class_counts.float())
        elif method == 'inverse':
            class_counts = torch.bincount(train_labels, minlength=n_classes)
            weights = 1.0 / class_counts.float()
            weights = weights / weights.sum() * n_classes
        else:
            raise ValueError(f"Unknown method: {method}")
        
        print(f"Class distribution: {torch.bincount(train_labels)}")
        print(f"Computed class weights ({method}): {weights}")
        
        return weights
        
class GrayScaleDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_name: Literal["cmnist"] = "cmnist",
        data_dir: str = "./assets/datasets",
        val_split_ratio: float = 0,
        batch_size: int = 256,
        resize: int = 28,
        num_workers: int = 1,
        channel_means: torch.Tensor = None,
        channel_stds: torch.Tensor = None,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.resize = resize
        self.num_workers = num_workers
        self.val_split_ratio = val_split_ratio
        # seed fixing the generator
        self.val_split_seed = 1
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.full_dataset = None
        self.test_labels = None
        self.val_labels = None
        # parameters for normalization and denormalization
        self.channel_means = channel_means
        self.channel_stds = channel_stds
        
        self.dataset_map = {
            "cmnist": CMNIST,
        }
        
        # parameters for normalization and denormalization
        self.channel_means = channel_means
        self.channel_stds = channel_stds
        self.normalize_fn = None

    def get_normalize_fn(self, data: torch.Tensor):
        self.channel_means = data.mean([0, 2, 3])
        self.channel_stds = data.std([0, 2, 3])
        return transforms.Normalize(self.channel_means, self.channel_stds)

    def denormalize_fn(self, tensor: torch.Tensor)->torch.Tensor:
        """This applies an inverse z-transformation to denormalize data
        """
        device = tensor.device
        return tensor.mul(self.channel_stds.to(device)).add(self.channel_means.to(device))
    
    def setup(self, stage: str = None):
        dataset_class = self.dataset_map[self.dataset_name]
        

        dataset_class(self.data_dir, train=False, transform=transforms.ToTensor(), download=True)
        
        if self.channel_means is None or self.channel_stds is None:
            # create temp loader to compute normalization statistics automatically
            temp_dataset = dataset_class(self.data_dir, train=True, transform=transforms.ToTensor(), download=True)
            temp_loader = DataLoader(temp_dataset, batch_size=10000, num_workers=self.num_workers)
            self.normalize_fn = self.get_normalize_fn(next(iter(temp_loader))[0])
        else:
            self.normalize_fn = transforms.Normalize(self.channel_means, self.channel_stds)
            
        
        transform_list = []
        if self.resize:
            transform_list.append(transforms.Resize((self.resize, self.resize)))
        transform_list.extend([transforms.ToTensor(), self.normalize_fn])
        base_transform = transforms.Compose(transform_list)
        
        aug_transform = transforms.Compose([
            *transform_list[:-2],
            transforms.RandomAffine(degrees=16, translate=(0.1, 0.1), shear=8),
            transforms.ToTensor(),
            self.normalize_fn
        ])


        base_test = dataset_class(self.data_dir, train=False, transform=None)
        
        base_train = dataset_class(self.data_dir, train=True, transform=None)
        if self.val_split_ratio > 0:
            base_train, base_val = get_train_val_split(base_train, self.val_split_seed, self.val_split_ratio)
        else:
            base_val = base_train
 
        self.train_dataset = MultiViewDataset(base_train, base_transform, aug_transform)
        self.val_dataset = MultiViewDataset(base_val, base_transform, base_transform)
        self.test_dataset = MultiViewDataset(base_test, base_transform, base_transform)
        self.full_dataset = torch.utils.data.ConcatDataset([self.train_dataset, self.test_dataset])
        self.test_labels = extract_labels_from_dataset(self.test_dataset)
        self.val_labels = extract_labels_from_dataset(self.val_dataset)

    def train_dataloader(self):
        dataset = self.train_dataset
        if "debug" in self.dataset_name:
            subsample_indices = torch.randperm(len(dataset))[:self.batch_size*2]
            sampler = SubsetRandomSampler(subsample_indices)
            shuffle = False
        else:
            sampler = None
            shuffle = True
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, sampler=sampler,
                         num_workers=self.num_workers, pin_memory=True, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                         num_workers=self.num_workers, pin_memory=True, persistent_workers=True)
    
    def test_dataloader(self):
        print("Run test")
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False,
                         num_workers=self.num_workers, pin_memory=True, persistent_workers=True)

     
