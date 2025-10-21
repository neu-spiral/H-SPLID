import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.models.utils import get_in_channels

class LeNet3(nn.Module):          
    '''
    Two convolutional layers of sizes 64 and 128, and a fully connected layer of size 1024
    suggested by 'Adversarial Robustness vs. Model Compression, or Both?'
    '''
    def __init__(self, feature_hook_fn=lambda x: x, **kwargs): 
        super().__init__()
        self.last_hidden_dim = kwargs["last_hidden_dim"]
        
        # NOTE: This is used to implement the feature splitting
        self.feature_hook_fn = feature_hook_fn
        
        ds_name = kwargs["dataset"]["name"]
        in_ch = kwargs["dataset"]["channels"]
        n_classes = kwargs["dataset"]["n_classes"]
        
        self.conv1 = torch.nn.Conv2d(in_ch, 32, 5, 1, 2) # in_channels, out_channels, kernel, stride, padding
        self.conv2 = torch.nn.Conv2d(32, 64, 5, 1, 2)
        
        # Fully connected layer
        if ds_name == "cmnist":
            dim = 16*16*64
        elif ds_name == 'coco-animals':
            dim = 56*56*64
            
        self.fc1 = torch.nn.Linear(dim, self.last_hidden_dim)   # convert matrix with 400 features to a matrix of 1024 features (columns)
        self.fc2 = torch.nn.Linear(self.last_hidden_dim, n_classes)
        
    def predict(self, z):
        z = self.feature_hook_fn(z)
        z = F.relu(z)
        return F.log_softmax(self.fc2(z), dim=1)
    
    def encode(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, np.prod(x.size()[1:]))
        x = self.fc1(x)
        return x
        
    def forward(self, x):
        x = self.encode(x)
        x = self.predict(x)
        return x
