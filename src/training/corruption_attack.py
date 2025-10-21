"""
Corruption-based attack implementation adapted from:
Hendrycks & Dietterich (2019) - https://github.com/hendrycks/robustness
Based on: make_imagenet_c.py
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import cv2
from skimage.filters import gaussian
import skimage as sk
from scipy.ndimage import zoom as scizoom
import warnings

warnings.simplefilter("ignore", UserWarning)

def safe_gaussian_blur(image, sigma):
    """
    Apply gaussian blur with compatibility for different scikit-image versions
    """
    try:
        # Try new API first
        return gaussian(image, sigma=sigma, channel_axis=-1)
    except TypeError:
        # Fall back to old API
        return gaussian(image, sigma=sigma, multichannel=True)
        

class CorruptionAttack:
    """
    Corruption-based attack that applies image corruptions.
    """
    
    def __init__(self, model, corruption_type='gaussian_noise', severity=1, 
                 attack_right=False, attack_protected=False):
        self.model = model
        self.corruption_type = corruption_type
        self.severity = severity
        self.attack_right = attack_right
        self.attack_protected = attack_protected
        if model is not None:
            self.device = next(model.parameters()).device
        else:
            self.device = torch.device('cpu')
        
        # Available corruption functions
        self.corruption_functions = {
            'gaussian_noise': self.gaussian_noise,
            'shot_noise': self.shot_noise,
            'impulse_noise': self.impulse_noise,
            'speckle_noise': self.speckle_noise,
            'gaussian_blur': self.gaussian_blur,
            'defocus_blur': self.defocus_blur,
            'zoom_blur': self.zoom_blur,
            'contrast': self.contrast,
            'brightness': self.brightness,
            'saturate': self.saturate,
            'snow': self.snow,
        }
        
        if corruption_type not in self.corruption_functions:
            raise ValueError(f"Corruption type {corruption_type} not supported. "
                           f"Available types: {list(self.corruption_functions.keys())}")
    
    def set_normalization_used(self, mean, std):
        """Set normalization parameters for denormalization/renormalization"""
        self.mean = torch.tensor(mean).view(1, -1, 1, 1).to(self.device)
        self.std = torch.tensor(std).view(1, -1, 1, 1).to(self.device)
    
    def denormalize(self, x):
        """Denormalize input tensor"""
        return x * self.std + self.mean
    
    def normalize(self, x):
        """Normalize input tensor"""
        return (x - self.mean) / self.std
    
    def prepare_mask(self, mask, target_shape):
        """
        Prepare mask for proper broadcasting with image tensor.
        
        Args:
            mask: Input mask tensor, could be [H, W], [1, H, W], or [3, H, W]
            target_shape: Target shape [C, H, W] to match
            
        Returns:
            Prepared mask tensor [C, H, W]
        """
        C, H, W = target_shape
        
        # Handle different mask dimensions
        if mask.dim() == 2:  # [H, W]
            mask = mask.unsqueeze(0)  # -> [1, H, W]
        
        if mask.dim() == 3:
            if mask.shape[0] == 1:  # [1, H, W]
                if C == 3:
                    mask = mask.repeat(3, 1, 1)  # -> [3, H, W]
            elif mask.shape[0] == 3:  # [3, H, W]
                pass  # Already correct
            elif mask.shape[0] == C:  # [C, H, W] - general case
                pass  # Already correct
            else:
                # If channels don't match, repeat the first channel
                mask = mask[0:1].repeat(C, 1, 1)
        else:
            raise ValueError(f"Unexpected mask dimensions: {mask.shape}")
        
        return mask.float()
    
    def __call__(self, inputs, labels, mask_protected=None):
        """
        Apply corruption attack to inputs
        
        Args:
            inputs: Input images tensor [B, C, H, W]
            labels: Target labels (not used for corruptions but kept for compatibility)
            mask_protected: Binary mask indicating protected areas [B, 1, H, W] or [B, H, W]
        
        Returns:
            Corrupted images tensor [B, C, H, W]
        """
        device = inputs.device
        inputs = inputs.clone().detach()
        
        # Denormalize inputs to [0, 1] range for corruption functions
        inputs_denorm = self.denormalize(inputs)
        inputs_denorm = torch.clamp(inputs_denorm, 0, 1)
        
        corrupted_batch = []
        
        for i in range(inputs.shape[0]):
            # Get single image
            img = inputs_denorm[i].cpu()  # [C, H, W]
            
            # Convert to numpy format [H, W, C] and scale to [0, 255]
            img_np = img.permute(1, 2, 0).numpy() * 255
            
            # Apply corruption
            corruption_fn = self.corruption_functions[self.corruption_type]
            corrupted_img = corruption_fn(img_np, self.severity)
            
            # Ensure corrupted_img is valid
            if corrupted_img is None:
                corrupted_img = img_np  # Fallback to original if corruption failed
            
            # Convert back to tensor format [C, H, W] and scale to [0, 1]
            corrupted_img = torch.from_numpy(corrupted_img / 255.0).permute(2, 0, 1).float()
            
            # Apply mask if provided
            if mask_protected is not None:
                mask = mask_protected[i].cpu()
                mask = self.prepare_mask(mask, img.shape)
                
                if self.attack_protected:
                    mask1 = (1 - mask).float()
                    mask2 = mask.float()
                else:
                    # Default case: attack everywhere
                    mask1 = torch.ones_like(mask).float()
                    mask2 = torch.zeros_like(mask).float()
                
                corrupted_img = corrupted_img * mask1 + img * mask2
            
            corrupted_batch.append(corrupted_img)
        
        # Stack batch and move to device
        corrupted_batch = torch.stack(corrupted_batch).to(device)
        
        # Renormalize for model input
        corrupted_batch = self.normalize(corrupted_batch)
        
        return corrupted_batch

    def gaussian_noise(self, x, severity=1):
        c = [.08, .12, 0.18, 0.26, 0.38][severity - 1]
        x = np.array(x) / 255.
        return np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255

    def shot_noise(self, x, severity=1):
        c = [60, 25, 12, 5, 3][severity - 1]
        x = np.array(x) / 255.
        return np.clip(np.random.poisson(x * c) / c, 0, 1) * 255

    def impulse_noise(self, x, severity=1):
        c = [.03, .06, .09, 0.17, 0.27][severity - 1]
        x = sk.util.random_noise(np.array(x) / 255., mode='s&p', amount=c)
        return np.clip(x, 0, 1) * 255

    def speckle_noise(self, x, severity=1):
        c = [.15, .2, 0.35, 0.45, 0.6][severity - 1]
        x = np.array(x) / 255.
        return np.clip(x + x * np.random.normal(size=x.shape, scale=c), 0, 1) * 255

    def gaussian_blur(self, x, severity=1):
        c = [1, 2, 3, 4, 6][severity - 1]
        x = np.array(x) / 255.
        x = safe_gaussian_blur(x, sigma=c)
        return np.clip(x, 0, 1) * 255

    def defocus_blur(self, x, severity=1):
        c = [(3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)][severity - 1]
        x = np.array(x) / 255.
        kernel = self.disk(radius=c[0], alias_blur=c[1])
        channels = []
        for d in range(3):
            channels.append(cv2.filter2D(x[:, :, d], -1, kernel))
        channels = np.array(channels).transpose((1, 2, 0))
        return np.clip(channels, 0, 1) * 255

    def zoom_blur(self, x, severity=1):
        c = [np.arange(1, 1.11, 0.01),
             np.arange(1, 1.16, 0.01),
             np.arange(1, 1.21, 0.02),
             np.arange(1, 1.26, 0.02),
             np.arange(1, 1.31, 0.03)][severity - 1]

        x = (np.array(x) / 255.).astype(np.float32)
        out = np.zeros_like(x)
        for zoom_factor in c:
            out += self.clipped_zoom(x, zoom_factor)
        x = (x + out) / (len(c) + 1)
        return np.clip(x, 0, 1) * 255

    def contrast(self, x, severity=1):
        c = [0.4, .3, .2, .1, .05][severity - 1]
        x = np.array(x) / 255.
        means = np.mean(x, axis=(0, 1), keepdims=True)
        return np.clip((x - means) * c + means, 0, 1) * 255

    def brightness(self, x, severity=1):
        c = [.1, .2, .3, .4, .5][severity - 1]
        x = np.array(x) / 255.
        x = sk.color.rgb2hsv(x)
        x[:, :, 2] = np.clip(x[:, :, 2] + c, 0, 1)
        x = sk.color.hsv2rgb(x)
        return np.clip(x, 0, 1) * 255

    def saturate(self, x, severity=1):
        c = [(0.3, 0), (0.1, 0), (2, 0), (5, 0.1), (20, 0.2)][severity - 1]
        x = np.array(x) / 255.
        x = sk.color.rgb2hsv(x)
        x[:, :, 1] = np.clip(x[:, :, 1] * c[0] + c[1], 0, 1)
        x = sk.color.hsv2rgb(x)
        return np.clip(x, 0, 1) * 255

    def snow(self, x, severity=1):
        """
        Snow corruption - using OpenCV motion blur instead of Wand
        """
        c = [(0.1, 0.3, 3, 0.5, 10, 4, 0.8),
             (0.2, 0.3, 2, 0.5, 12, 4, 0.7),
             (0.55, 0.3, 4, 0.9, 12, 8, 0.7),
             (0.55, 0.3, 4.5, 0.85, 12, 8, 0.65),
             (0.55, 0.3, 2.5, 0.85, 12, 12, 0.55)][severity - 1]

        x = np.array(x, dtype=np.float32) / 255.
        h, w = x.shape[:2]
        
        # Generate snow layer - monochrome noise
        snow_layer = np.random.normal(size=x.shape[:2], loc=c[0], scale=c[1])  # [:2] for monochrome

        # Apply zoom to the snow layer
        snow_layer = self.clipped_zoom(snow_layer[..., np.newaxis], c[2])
        
        # Threshold the snow layer
        snow_layer[snow_layer < c[3]] = 0
        snow_layer = snow_layer.squeeze()

        # Apply motion blur using OpenCV
        # Create motion blur kernel
        angle = np.random.uniform(-135, -45)  # Random angle
        kernel_size = int(c[4])  # Use radius parameter for kernel size
        kernel = self.create_motion_blur_kernel(kernel_size, angle)
        
        # Apply motion blur
        snow_layer = cv2.filter2D(snow_layer.astype(np.float32), -1, kernel)
        
        # Normalize and add dimension
        if np.max(snow_layer) > 0:
            snow_layer = snow_layer / np.max(snow_layer)
        snow_layer = snow_layer[..., np.newaxis]

        # Apply the snow effect
        x = c[6] * x + (1 - c[6]) * np.maximum(x, cv2.cvtColor(x, cv2.COLOR_RGB2GRAY).reshape(h, w, 1) * 1.5 + 0.5)
        
        # Add snow layers with rotation
        result = x + snow_layer + np.rot90(snow_layer, k=2, axes=(0, 1))
        
        return np.clip(result, 0, 1) * 255

    def create_motion_blur_kernel(self, size, angle):
        """
        Create a motion blur kernel using OpenCV
        """
        # Convert angle to radians
        angle = np.deg2rad(angle)
        
        kernel = np.zeros((size, size), dtype=np.float32)
        
        # Calculate the line coordinates
        center = size // 2
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        
        # Draw line in kernel
        for i in range(size):
            offset = i - center
            x = int(center + offset * cos_angle)
            y = int(center + offset * sin_angle)
            
            # Ensure coordinates are within bounds
            if 0 <= x < size and 0 <= y < size:
                kernel[y, x] = 1.0
        
        # Normalize kernel
        if np.sum(kernel) > 0:
            kernel = kernel / np.sum(kernel)
        
        return kernel

    # Helper functions
    def disk(self, radius, alias_blur=0.1, dtype=np.float32):
        if radius <= 8:
            L = np.arange(-8, 8 + 1)
            ksize = (3, 3)
        else:
            L = np.arange(-radius, radius + 1)
            ksize = (5, 5)
        X, Y = np.meshgrid(L, L)
        aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
        aliased_disk /= np.sum(aliased_disk)
        return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)

    def clipped_zoom(self, img, zoom_factor):
        h = img.shape[0]
        ch = int(np.ceil(h / zoom_factor))
        top = (h - ch) // 2
        img = scizoom(img[top:top + ch, top:top + ch], (zoom_factor, zoom_factor, 1), order=1)
        trim_top = (img.shape[0] - h) // 2
        return img[trim_top:trim_top + h, trim_top:trim_top + h]