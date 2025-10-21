import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
# Imports for our backbone models
from src.training.utils import set_optimizer
from sklearn.metrics import accuracy_score
from src.training.utils import get_attack_modality

import torch
import torch.nn as nn
import pytorch_lightning as pl

class StandardClassificationModel(pl.LightningModule):
    def __init__(
        self, 
        backbone_model: nn.Module,
        **kwargs,
    ):
        super().__init__()
        self.model = backbone_model
        self.kwargs = kwargs
        self.sparsity_lambda = 0 if "sparsity_lambda" not in self.kwargs.keys() else self.kwargs["sparsity_lambda"]
        self.sparsity_type = None if "sparsity_type" not in self.kwargs.keys() else self.kwargs["sparsity_type"]
        
        # Class weighting configuration
        self.use_class_weights = kwargs.get("use_class_weights", False)
        self.class_weight_method = kwargs.get("class_weight_method", "balanced")
        self.class_weights = None
        
        # needs to be set to false, such that gradient clipping can be used
        self.automatic_optimization = False
        
        model_type = kwargs.get('model', '').lower()
    
        if "lenet" in model_type:
            # LeNet: use fc1 layer
            penultimate_layer = self.model.fc1
        elif "resnet" in model_type:
            # ResNet: use avgpool layer
            penultimate_layer = self.model.avgpool
        else:
            raise ValueError(f"Unsupported model type: {model_type}. Supported models are: LeNet, ResNet.")
        
        if penultimate_layer is not None:
            penultimate_layer.register_forward_hook(self.hook_fn)
        
        
        self.save_hyperparameters(ignore=['backbone_model'], logger=False)

    def on_fit_start(self):
        """Initialize class weights when training starts"""
        if self.use_class_weights and self.class_weights is None:
            try:
                # Get class weights from datamodule
                weights = self.trainer.datamodule.compute_class_weights(method=self.class_weight_method)
                self.class_weights = weights.to(self.device)
                print(f"Initialized class weights: {self.class_weights}")
            except Exception as e:
                print(f"Warning: Could not compute class weights: {e}")
                print("Falling back to uniform weights")
                # Get number of classes from the model or config
                n_classes = self.kwargs.get('dataset', {}).get('n_classes', 3)  # Default to 3 for ISIC2017
                self.class_weights = torch.ones(n_classes).to(self.device)

    def compute_loss(self, prediction, target):
        """Compute loss with optional class weighting"""
        if self.use_class_weights and self.class_weights is not None:
            return F.cross_entropy(prediction, target, weight=self.class_weights)
        else:
            return F.cross_entropy(prediction, target)
    
    def forward(self, x):
        output = self.model(x)
        # NOTE: Depending on the modus of some models, they can return a tuple of 2 values              
        if isinstance(output, tuple):
            if len(output) == 2:
                prediction = output[0]
        else:
            prediction = output
        return prediction

    def _forward_step(self, batch, batch_idx, mode):
        if len(batch) == 2:
            data_aug, target = batch
            data_orig = None
            masks = None
        elif len(batch) == 3:
            data_aug, data_orig, target = batch
        elif len(batch) == 4: # dataloader: data, target, masks (e.g., coco)
            data_aug, data_orig, target, _masks = batch 
        else:
            raise ValueError(f"Unexpected batch size of {len(batch)} encountered")
        # NOTE: For val and test set, data_aug is the same as data_orig
        prediction = self(data_aug)
        loss = self.compute_loss(prediction, target)

        prediction_labels = prediction.argmax(1).detach().cpu().numpy()
        accuracy = accuracy_score(prediction_labels, target.detach().cpu().numpy())*100

        # Logging
        self.log(f'{mode}/ce_loss', loss, prog_bar=True)

        return loss, accuracy
    
    def hook_fn(self, module, input, output):
        """Store penultimate layer activations."""
        if len(output.shape) == 4:
            # CNN layers (ResNet avgpool): [B, C, H, W] -> [B, C]
            self.penultimate_activation = torch.flatten(output, 1)
        elif len(output.shape) == 2:
            # Already flattened (LeNet fc1): [B, features]
            self.penultimate_activation = output
        else:
            raise ValueError(f"Unexpected output shape: {output.shape}")
    
    def sparsity_loss(self):
        """Compute the sparsity penalty."""
        if self.sparsity_type is None:
            return torch.tensor(0.0, device=self.device)
        if "activation" in self.sparsity_type.lower():
            return self.penultimate_layer_sparsity_loss()
        elif "weights" in self.sparsity_type.lower():
            return self.weight_sparsity_loss()

    def penultimate_layer_sparsity_loss(self):
        """Compute the sparsity penalty for penultimate_layer."""
        if self.penultimate_activation is None:
            return torch.tensor(0.0, device=self.device)

        if self.sparsity_type == "L1-activation":
            return torch.norm(self.penultimate_activation, p=1)  # L1 regularization
        elif self.sparsity_type == "group_lasso-activation":
            return torch.sum(torch.norm(self.penultimate_activation, p=2, dim=1))  # Group Lasso
        
        return torch.tensor(0.0, device=self.device)
    
    
    def weight_sparsity_loss(self):
        """
        Compute the sparsity penalty for model weights.
        - L1 sparsity: element-wise sparsity
        - Group Lasso: structured sparsity across filters/neuron groups
        """
        sparsity_loss = 0.0
        for name, param in self.model.named_parameters():
            if "weight" in name and param.requires_grad:
                # Exclude BatchNorm layers
                if any(layer_type in name.lower() for layer_type in ["bn", "batchnorm"]):
                    continue
                if self.sparsity_type == "L1-weights":
                    sparsity_loss += torch.norm(param, p=1)  # L1 norm
                elif self.sparsity_type == "group_lasso-weights":
                    if param.dim() == 4:  # Convolutional Layer (4D)
                        sparsity_loss += torch.sum(torch.norm(param, p=2, dim=(1, 2, 3)))  # Sum over filters
                    elif param.dim() == 2:  # Fully Connected Layer (2D)
                        sparsity_loss += torch.sum(torch.norm(param, p=2, dim=1))  # Sum over neurons
                
        return sparsity_loss
    
    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        
        # compute loss
        loss, accuracy = self._forward_step(batch, batch_idx, mode="train")
        
        sparsity_loss = self.sparsity_loss()
        loss += self.sparsity_lambda*sparsity_loss
        
        opt.zero_grad()
        # calculate gradients
        self.manual_backward(loss)
        
        # Gradient clipping
        self.clip_gradients(opt, gradient_clip_val=1000, gradient_clip_algorithm="norm")

        # compute norms for logging
        l1_grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1e16, norm_type=1.0)
        l2_grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1e16, norm_type=2.0)
        
        # update network
        opt.step()
        
        # perform scheduler step
        sch = self.lr_schedulers()
        if sch is not None:
            sch.step()
        
        # Logging
        self.log('train/total_loss', loss, prog_bar=True)
        self.log('train/ACC', accuracy, prog_bar=True)
        self.log('train/l1_grad_norm', l1_grad_norm)
        self.log('train/l2_grad_norm', l2_grad_norm)
        if self.sparsity_type is not None:
            self.log(f'train/{self.sparsity_type.upper()}_sparsity_loss', sparsity_loss)
        self.log(f'train/non_zero_features', (self.penultimate_activation.mean(0) > 0).sum().float())
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, accuracy = self._forward_step(batch, batch_idx, mode="val")
        self.log('val/total_loss', loss, prog_bar=True)
        self.log('val/ACC', accuracy, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, accuracy = self._forward_step(batch, batch_idx, mode="test")
        self.log('test/total_loss', loss, prog_bar=True)
        self.log('test/ACC', accuracy, prog_bar=True)
        return loss
    
    # def on_train_epoch_end(self):
    #     pass

    def configure_optimizers(self):
        # code to access trainloader in pytorch-lightning
        self.trainer.fit_loop.setup_data()
        dataloader = self.trainer.train_dataloader
        # NOTE: Scheduler can be None if no scheduler is defined in kwargs
        optimizer, scheduler = set_optimizer(self.kwargs, self.model, dataloader)
        if scheduler is None:
            return [optimizer]
        else:
            return [optimizer], [scheduler]

def get_attacked_model_prediction(pl_module, data, target, mask, **kwargs):
    if kwargs['attack']["attack_protected"] and mask is None:
        raise ValueError("Mask cannot be None if 'attack.attack_protected is True")
    attack = get_attack_modality(pl_module.model, **kwargs)
    # Denormalize data such that it is between 0 and 1. This is important as torchattacks assumes data
    # to be between 0 and 1
    means = kwargs["dataset"]["channel_means"]
    stds = kwargs["dataset"]["channel_stds"]   
    attack.set_normalization_used(means,stds)
    # Enable grad needs to be used here, otherwise validation and testing does not work,
    # see: https://github.com/Lightning-AI/pytorch-lightning/discussions/14782
    with torch.enable_grad():
        if mask is not None:
            # create 3-channel masks
            if kwargs["dataset"]["name"] != "masked-mnist":
                mask = mask.repeat(1, 3, 1, 1)
            mask = mask.to(pl_module.device)
        attacked_data = attack(data, target, mask_protected=mask)
    
    # use attacked data
    prediction = pl_module(attacked_data)
    return prediction, attacked_data
