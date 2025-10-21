import json
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Set

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader


def load_synset_index(meta_file: Path) -> Dict[str, int]:
    """
    Load ImageNet synset-to-index mapping from a JSON file.
    """
    if not meta_file.is_file():
        raise FileNotFoundError(f"Missing metadata file: {meta_file}")
    data = json.loads(meta_file.read_text())
    return {syn: int(idx) for idx, (syn, _) in data.items()}


def extract_prefix(stem: str, subset: str) -> str:
    """
    Extract synset prefix from a filename stem based on subset rules.
    - mixed_rand: expects 'fg_<synset>_...'
    - others: '<synset>_...' or '<synset>'
    """
    if subset == 'mixed_rand':
        if not stem.startswith('fg_'):
            raise ValueError(f"Invalid filename for mixed_rand: '{stem}'")
        parts = stem.split('_', 2)
        if len(parts) < 2 or not parts[1]:
            raise ValueError(f"Cannot parse synset from: '{stem}'")
        return parts[1]

    return stem.split('_', 1)[0]


def gather_samples(
    root: Path,
    subset: str,
    split: str,
    exts: Set[str],
    syn2idx: Dict[str, int]
) -> List[Tuple[str, int]]:
    """
    Traverse dataset directory and collect valid sample paths with their indices.
    """
    base = root / subset / split
    if not base.is_dir():
        raise FileNotFoundError(f"Dataset path not found: {base}")

    samples = []
    for class_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        for file in sorted(class_dir.iterdir()):
            if not file.is_file():
                continue
            suffix = file.suffix.lower()
            if suffix not in exts and suffix != '.npy':
                continue
            stem = file.stem
            prefix = extract_prefix(stem, subset)
            idx = syn2idx.get(prefix)
            if idx is not None:
                samples.append((str(file), idx))
    if not samples:
        raise ValueError(f"No samples found under {base}")
    return samples


class ImageNet9(Dataset):
    """
    Functional-style PyTorch Dataset for ImageNet-9.
    """
    ALLOWED_SUBSETS = {'original', 'only_fg', 'mixed_rand'}
    IMAGE_EXTS = {'.jpg', '.jpeg', '.png'}

    def __init__(
        self,
        root: str,
        subset: str = 'original',
        split: str = 'val',
        transform: Callable = None
    ):
        if subset not in self.ALLOWED_SUBSETS:
            raise ValueError(f"Unsupported subset '{subset}'. Choose from {self.ALLOWED_SUBSETS}.")

        self.root = Path(root)
        self.subset = subset
        self.split = split
        self.transform = transform

        meta_file = self.root / 'imagenet_class_index.json'
        syn2idx = load_synset_index(meta_file)

        exts = self.IMAGE_EXTS
        self.samples = gather_samples(self.root, subset, split, exts, syn2idx)
        self.targets = [idx for _, idx in self.samples]

        print(f"Loaded {len(self.samples)} samples from '{subset}/{split}'")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path, target = self.samples[idx]
        suffix = Path(path).suffix.lower()
        data = np.load(path) if suffix == '.npy' else default_loader(path)
        if self.transform:
            data = self.transform(data)
        return data, torch.tensor(target, dtype=torch.long)
