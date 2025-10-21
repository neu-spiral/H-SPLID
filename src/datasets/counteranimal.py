import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from typing import Literal, Union

class CounterAnimal(Dataset):
    def __init__(
        self,
        root: Union[str, Path],
        mode: Literal["common", "counter", "both"] = "both",
        transform=None
    ):
        """
        A dataset for evaluating on the CounterAnimal benchmark.

        Args:
            root: Path to the CounterAnimal dataset root, which should contain per-class
                  subdirectories named like "0 cat", "1 dog", etc. Each of those
                  directories must contain two subfolders: "common-<class_name>"
                  and "counter-<class_name>".
            mode: One of 'common', 'counter', or 'both'. Selects which subset of images
                  to load for evaluation.
            transform: A torchvision transform (or composition) to apply to the images.
        """
        super().__init__()
        self.root = Path(root) / "LAION-final"
        assert mode in ("common", "counter", "both"), \
            "mode must be 'common', 'counter', or 'both'"
        self.mode = mode
        self.transform = transform
        self.samples = []  # list of (image_path, label)

        # Walk through each class folder
        for class_dir in sorted(self.root.iterdir()):
            if not class_dir.is_dir():
                continue

            # 1) extract label (first token) and ignore the rest of the folder-name whitespace
            parts = class_dir.name.split(None, 1)   # split on any whitespace, max 1 split
            try:
                label = int(parts[0])
            except ValueError:
                continue

            # 2) now just find ANY subfolder that starts with "common-" or "counter-"
            subdirs = []
            if self.mode in ("common", "both"):
                subdirs += [d for d in class_dir.iterdir()
                            if d.is_dir() and d.name.startswith("common-")]
            if self.mode in ("counter", "both"):
                subdirs += [d for d in class_dir.iterdir()
                            if d.is_dir() and d.name.startswith("counter-")]

            # 3) collect images
            for subdir in sorted(subdirs):
                for img_file in sorted(subdir.iterdir()):
                    if img_file.is_file() and img_file.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                        self.samples.append((str(img_file), label))

        # Optionally provide a targets attribute for compatibility
        self.targets = [lbl for _, lbl in self.samples]
        print(f"Loaded {len(self.samples)} samples from {self.root} with mode '{self.mode}'")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label)
