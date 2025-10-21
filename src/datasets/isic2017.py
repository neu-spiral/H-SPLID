import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision
from pathlib import Path
import pickle
import hashlib
from concurrent.futures import ThreadPoolExecutor
import warnings
from tqdm import tqdm

class ISIC2017(Dataset):
    """ISIC 2017 Dataset"""

    def __init__(
        self, 
        root, 
        train=True, 
        transform=None, 
        target_transform=None,
        download=False,
        # Optimization parameters
        cache_mode='memory',  # 'memory', 'disk', 'none'
        cache_dir=None,
        preprocess=True,
        target_size=(224, 224),
        num_workers=4,
        memory_limit_gb=8,
        force_refresh=False
    ):
        """
        Args:
            root (str): Root directory containing the ISIC 2017 dataset
            train (bool): If True, load training set. If False, load test set
            transform (callable, optional): Transform for images
            target_transform (callable, optional): Transform for masks
            download (bool): Not used, but kept for interface compatibility
            cache_mode (str): 'memory' (RAM), 'disk' (preprocessed files), or 'none'
            cache_dir (str): Directory for disk cache (defaults to root/cache)
            preprocess (bool): Whether to preprocess images to target_size
            target_size (tuple): Target size for preprocessing
            num_workers (int): Number of workers for parallel preprocessing
            memory_limit_gb (float): Memory limit for caching in GB
            force_refresh (bool): Force refresh of cache
        """
        self.root = Path(root) / 'isic2017'
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.target_size = target_size
        self.cache_mode = cache_mode
        self.num_workers = num_workers
        self.force_refresh = force_refresh
        
        # Setup cache directory
        if cache_dir is None:
            self.cache_dir = self.root / 'cache'
        else:
            self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Memory limit in bytes
        self.memory_limit = memory_limit_gb * 1024**3
        
        if train:
            self.splits = ['train', 'val']
        else:
            self.splits = ['test']
        
        # Initialize data containers
        self.image_paths = []
        self.mask_paths = []
        self.labels = []
        self.image_ids = []
        
        # Cache containers
        self.image_cache = {}
        self.mask_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        self.cache_as_tensors = False
        
        if preprocess:
            self.default_image_transform = torchvision.transforms.Resize(self.target_size)
            self.default_mask_transform = torchvision.transforms.Resize(self.target_size)
        else:
            self.default_image_transform = None
            self.default_mask_transform = None
        
        # Load metadata first
        for split in self.splits:
            self._load_split_data(split)
        
        print(f"Loaded {len(self.image_paths)} samples")
        
        # Initialize caching based on mode
        if self.cache_mode == 'memory':
            self._init_memory_cache()
        elif self.cache_mode == 'disk':
            self._init_disk_cache()
    
    def _get_cache_key(self):
        """Generate cache key based on dataset configuration"""
        config_str = f"{self.train}_{self.target_size}_{len(self.image_paths)}"
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def _init_memory_cache(self):
        """Initialize memory cache by preloading all data"""
        cache_file = self.cache_dir / f"memory_cache_{self._get_cache_key()}.pkl"
        
        if cache_file.exists() and not self.force_refresh:
            print("Loading from memory cache file...")
            try:
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    self.image_cache = cache_data['images']
                    self.mask_cache = cache_data['masks']
                print(f"Loaded {len(self.image_cache)} items from cache")
                return
            except Exception as e:
                print(f"Failed to load cache: {e}. Rebuilding...")
        
        # Estimate memory usage (rough estimate for PIL Images)
        estimated_memory = len(self.image_paths) * np.prod(self.target_size) * 3 * 1  # RGB uint8
        estimated_memory += len(self.image_paths) * np.prod(self.target_size) * 1     # mask uint8
        
        if estimated_memory > self.memory_limit:
            warnings.warn(f"Estimated memory usage ({estimated_memory/1024**3:.1f}GB) exceeds limit ({self.memory_limit/1024**3:.1f}GB). Consider using disk cache or reducing target_size.")
            self.cache_mode = 'disk'
            self._init_disk_cache()
            return
        
        print("Preprocessing and caching all data in memory...")
        self._preload_all_data()
        
        # Save cache to disk for future use
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'images': self.image_cache,
                    'masks': self.mask_cache
                }, f)
            print(f"Saved cache to {cache_file}")
        except Exception as e:
            print(f"Failed to save cache: {e}")
    
    def _init_disk_cache(self):
        """Initialize disk cache by preprocessing data to disk"""
        cache_subdir = self.cache_dir / f"disk_cache_{self._get_cache_key()}"
        cache_subdir.mkdir(exist_ok=True)
        
        # Check if cache exists
        cache_info_file = cache_subdir / "cache_info.pkl"
        if cache_info_file.exists() and not self.force_refresh:
            with open(cache_info_file, 'rb') as f:
                cache_info = pickle.load(f)
                if len(cache_info['files']) == len(self.image_paths):
                    print(f"Using existing disk cache with {len(cache_info['files'])} files")
                    self.disk_cache_dir = cache_subdir
                    return
        
        print("Creating disk cache...")
        self._create_disk_cache(cache_subdir)
        self.disk_cache_dir = cache_subdir
    
    def _preload_all_data(self):
        """Preload all data into memory using parallel processing"""
        def process_item(idx):
            try:
                image_path = self.image_paths[idx]
                mask_path = self.mask_paths[idx]
                
                # Load and preprocess image
                image = Image.open(image_path).convert('RGB')
                mask = Image.open(mask_path).convert('L')
                mask = self._threshold_mask(mask)
                
                # Apply preprocessing transforms if specified
                if self.default_image_transform is not None:
                    image = self.default_image_transform(image)
                if self.default_mask_transform is not None:
                    mask = self.default_mask_transform(mask)
                
                return idx, image, mask
            except Exception as e:
                print(f"Error processing item {idx}: {e}")
                return idx, None, None
        
        # Use ThreadPoolExecutor for I/O bound operations
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(tqdm(
                executor.map(process_item, range(len(self.image_paths))),
                total=len(self.image_paths),
                desc="Loading data"
            ))
        
        # Store in cache
        for idx, image, mask in results:
            if image is not None and mask is not None:
                self.image_cache[idx] = image
                self.mask_cache[idx] = mask
    
    def _create_disk_cache(self, cache_dir):
        """Create disk cache by preprocessing all data"""
        def process_and_save(idx):
            try:
                image_path = self.image_paths[idx]
                mask_path = self.mask_paths[idx]
                
                # Load and preprocess
                image = Image.open(image_path).convert('RGB')
                mask = Image.open(mask_path).convert('L')
                mask = self._threshold_mask(mask)
                
                # Apply preprocessing transforms if specified
                if self.default_image_transform is not None:
                    image = self.default_image_transform(image)
                if self.default_mask_transform is not None:
                    mask = self.default_mask_transform(mask)
                
                # Save to disk as PIL Images
                cache_file = cache_dir / f"item_{idx:06d}.pkl"
                with open(cache_file, 'wb') as f:
                    pickle.dump({
                        'image': image,
                        'mask': mask,
                        'label': self.labels[idx],
                        'image_id': self.image_ids[idx],
                        'cached_as_tensors': self.cache_as_tensors
                    }, f)
                
                return cache_file.name
            except Exception as e:
                print(f"Error processing item {idx}: {e}")
                return None
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            cache_files = list(tqdm(
                executor.map(process_and_save, range(len(self.image_paths))),
                total=len(self.image_paths),
                desc="Creating disk cache"
            ))
        
        # Save cache info
        cache_info = {
            'files': [f for f in cache_files if f is not None],
            'target_size': self.target_size,
            'num_items': len(self.image_paths),
            'cached_as_tensors': self.cache_as_tensors
        }
        
        with open(cache_dir / "cache_info.pkl", 'wb') as f:
            pickle.dump(cache_info, f)
    
    def _threshold_mask(self, mask, threshold=127):
        mask_array = np.array(mask, dtype=np.uint8)
        binary_mask = (mask_array > threshold).astype(np.uint8) * 255
        return Image.fromarray(binary_mask, mode='L')
    
    def _load_split_data(self, split):
        """Load data from a specific split"""
        split_dir = self.root / split
        
        if split == 'train':
            images_dir = split_dir / 'images' / 'ISIC-2017_Training_Data'
            masks_dir = split_dir / 'masks' / 'ISIC-2017_Training_Part1_GroundTruth'
            csv_file = 'ISIC-2017_Training_Part3_GroundTruth.csv'
        elif split == 'val':
            images_dir = split_dir / 'images' / 'ISIC-2017_Validation_Data'
            masks_dir = split_dir / 'masks' / 'ISIC-2017_Validation_Part1_GroundTruth'
            csv_file = 'ISIC-2017_Validation_Part3_GroundTruth.csv'
        elif split == 'test':
            images_dir = split_dir / 'images' / 'ISIC-2017_Test_v2_Data'
            masks_dir = split_dir / 'masks' / 'ISIC-2017_Test_v2_Part1_GroundTruth'
            csv_file = 'ISIC-2017_Test_v2_Part3_GroundTruth.csv'
        else:
            raise ValueError(f"Invalid split: {split}")
        
        csv_path = split_dir / csv_file
        
        if not csv_path.exists():
            raise FileNotFoundError(f"Labels CSV not found: {csv_path}")
        
        labels_df = pd.read_csv(csv_path)
        
        for _, row in labels_df.iterrows():
            image_id = row['image_id']
            
            # Find image file
            image_path = images_dir / f"{image_id}.jpg"
            if not image_path.exists():
                image_files = list(images_dir.glob(f"{image_id}*"))
                if image_files:
                    image_path = image_files[0]
                else:
                    print(f"Warning: Image not found for {image_id}, skipping")
                    continue
            
            # Find mask file
            mask_path = masks_dir / f"{image_id}_segmentation.png"
            if not mask_path.exists():
                mask_files = list(masks_dir.glob(f"{image_id}*"))
                if mask_files:
                    mask_path = mask_files[0]
                else:
                    print(f"Warning: Mask not found for {image_id}, skipping")
                    continue
            
            melanoma = float(row['melanoma'])
            keratosis = float(row['seborrheic_keratosis'])

            # Create multi-class label (0: nevus, 1: melanoma, 2: seborrheic_keratosis)
            if melanoma == 1.0:
                class_label = 1
            elif keratosis == 1.0:
                class_label = 2
            else:
                class_label = 0
            
            # Add to lists
            self.image_paths.append(image_path)
            self.mask_paths.append(mask_path)
            self.labels.append(class_label)
            self.image_ids.append(image_id)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Memory cache
        if self.cache_mode == 'memory':
            if idx in self.image_cache:
                self.cache_hits += 1
                image = self.image_cache[idx]
                mask = self.mask_cache[idx]
                label = self.labels[idx]
                
                # Apply additional transforms if specified
                if self.transform is not None:
                    image = self.transform(image)
                
                if self.target_transform is not None:
                    mask = self.target_transform(mask)
                
                return image, label, mask
        
        # Disk cache
        elif self.cache_mode == 'disk':
            cache_file = self.disk_cache_dir / f"item_{idx:06d}.pkl"
            if cache_file.exists():
                self.cache_hits += 1
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                image = cached_data['image']
                mask = cached_data['mask']
                label = cached_data['label']
                
                # Apply additional transforms if specified
                if self.transform is not None:
                    image = self.transform(image)
                
                if self.target_transform is not None:
                    mask = self.target_transform(mask)
                
                return image, label, mask
        
        # Fallback: load from disk
        self.cache_misses += 1
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        mask_path = self.mask_paths[idx]
        mask = Image.open(mask_path).convert('L')
        mask = self._threshold_mask(mask)
        
        label = self.labels[idx]
        
        # Apply preprocessing transforms
        if self.default_image_transform is not None:
            image = self.default_image_transform(image)
        if self.default_mask_transform is not None:
            mask = self.default_mask_transform(mask)
        
        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)
        
        if self.target_transform is not None:
            mask = self.target_transform(mask)
        
        return image, label, mask
    
    def get_cache_stats(self):
        """Get cache performance statistics"""
        total_accesses = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_accesses if total_accesses > 0 else 0
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'cache_mode': self.cache_mode
        }
    
    def clear_cache(self):
        """Clear cache and reset statistics"""
        self.image_cache.clear()
        self.mask_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0