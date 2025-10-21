"""
Code referenced from Wang et al. (2021)
https://github.com/neu-spiral/HBaR
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.hsic import hsic_objective
from src.models import StandardClassificationModel
from sklearn.metrics import accuracy_score

class HBaR(StandardClassificationModel):
    def __init__(self, backbone_model: nn.Module, **kwargs):
        super().__init__(backbone_model=backbone_model, **kwargs)
        try:
            backbone_model.feature_hook_fn
        except AttributeError:
            backbone_model.feature_hook_fn = lambda x: x
        self.model = backbone_model
        self.kwargs = kwargs
        self.include_hsic = 0
        
        # hbar-specific parameters (adjust in hbar_config.yaml)
        self.xentropy_weight = self.kwargs["hbar_hps"]["xentropy_weight"]
        self.smooth = self.kwargs["hbar_hps"]["smooth"]
        self.smooth_eps = self.kwargs["hbar_hps"]["smooth_eps"]
        self.lambda_x = self.kwargs["hbar_hps"]["lambda_x"]
        self.lambda_y = self.kwargs["hbar_hps"]["lambda_y"]
        self.sigma = self.kwargs["hbar_hps"]["sigma"]
        self.k_type_y = self.kwargs["hbar_hps"]["k_type_y"]
        self.hsic_layer_decay = self.kwargs["hbar_hps"]["hsic_layer_decay"]
        self.save_hyperparameters(ignore=['backbone_model'], logger=False)
        self.n_classes = self.kwargs["dataset"]["n_classes"]

    def _forward_step(self, batch, batch_idx, mode):
        """
        computes one forward step of a training batch
        """
        total_loss = 0.0
        prec1 = total_loss = hx_l = hy_l = -1   
        criterion = CrossEntropyLossMaybeSmooth(smooth_eps=self.smooth_eps).to(self.device) # option to add label smoothing from hbar

        if len(batch) == 3:
            data_aug, data_orig, target = batch
            masks = None
        elif len(batch) == 4: # dataloader: data, target, masks (e.g., coco)
            data_aug, data_orig, target, masks = batch 
        else:
            raise ValueError(f"Unexpected batch size of {len(batch)} encountered")
        
        # extract embeddings
        emb, hiddens = self.model.encode(data_aug, hb=True)

        # extract logits
        prediction = self.model.predict(emb)

        loss = criterion(prediction, target, smooth=self.smooth)
        total_loss += (loss * self.xentropy_weight)

        # compute hsic
        h_target = target.view(-1,1).to(self.device)
        h_target = to_categorical(h_target, num_classes=self.n_classes).float()
        h_data = data_aug.view(-1, np.prod(data_aug.size()[1:]))

        # hsic loss (summed up over layers)
        hx_l_list = []
        hy_l_list = []
        lx, ly, ld = self.lambda_x, self.lambda_y, self.hsic_layer_decay
        if ld > 0:
            lx, ly = lx * (ld ** len(hiddens)), ly * (ld ** len(hiddens))
            
        for i in range(len(hiddens)):
            
            if len(hiddens[i].size()) > 2:
                hiddens[i] = hiddens[i].view(-1, np.prod(hiddens[i].size()[1:]))

            hx_l, hy_l = hsic_objective(
                    hiddens[i],
                    h_target=h_target.float(),
                    h_data=h_data,
                    sigma=self.sigma,
                    k_type_y=self.k_type_y
            )

            hx_l_list.append(hx_l)
            hy_l_list.append(hy_l)
            
            if ld > 0:
                lx, ly = lx/ld, ly/ld
            temp_hsic = lx * hx_l - ly * hy_l
            total_loss += temp_hsic.to(self.device)

        # evaluate
        prediction_labels = prediction.argmax(1).detach().cpu().numpy()
        accuracy = accuracy_score(prediction_labels, target.detach().cpu().numpy()) * 100 

        # logging 
        self.log(f'HBaR/Total_Loss', total_loss)
        self.log(f'HBaR/CE_loss, scaling: {self.xentropy_weight}', loss * self.xentropy_weight)
        self.log(f'HBaR/hsic(x, z), scaling: {self.lambda_x}', hx_l * lx)
        self.log(f'HBaR/hsic(y, z), scaling: {self.lambda_y}', hy_l * ly)
        self.log(f'HBaR/Batch Acc', accuracy)
        
        return total_loss, accuracy 
    
"""
HBaR tricks 
"""
class CrossEntropyLossMaybeSmooth(nn.CrossEntropyLoss):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    def __init__(self, smooth_eps=0.0):
        super(CrossEntropyLossMaybeSmooth, self).__init__()
        self.smooth_eps = smooth_eps

    def forward(self, output, target, smooth=False):
        if not smooth:
            return F.cross_entropy(output, target)

        target = target.contiguous().view(-1)
        n_class = output.size(1)
        one_hot = torch.zeros_like(output).scatter(1, target.view(-1, 1), 1)
        smooth_one_hot = one_hot * (1 - self.smooth_eps) + (1 - one_hot) * self.smooth_eps / (n_class - 1)
        log_prb = F.log_softmax(output, dim=1)
        loss = -(smooth_one_hot * log_prb).sum(dim=1).mean()
        return loss


def to_categorical(y, num_classes):
    ''' 1-hot encodes a tensor '''
    return torch.squeeze(torch.eye(num_classes, device=y.device)[y])