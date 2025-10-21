import numpy as np
import torch
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score
from src.models import get_attacked_model_prediction
from src.training.utils import compute_distances_between_images
import random

class RobustnessMetricsCallback(pl.Callback):
    def __init__(self, data, add_hook_fn=None, panel_name="", **kwargs,):
        super().__init__()
        self.data = data
        self.kwargs = kwargs
        self.log_interval = self.kwargs["log_interval"]
        self.panel_name = panel_name
        self.add_hook_fn = add_hook_fn
        
    def on_train_epoch_end(self, trainer, pl_module):
        
        with torch.no_grad():
            if (trainer.current_epoch == 0 or (trainer.current_epoch % self.log_interval == 0) or (trainer.current_epoch+1 == trainer.max_epochs)):
                pl_module.eval()
                robust_accuracy, dist_dict = robustness_evaluation(pl_module, self.data, self.kwargs, return_dist_dict=True)
                if self.kwargs["attack"]["attack_right"] or self.kwargs["attack"]["attack_protected"]:
                    # create copies
                    copy_right = self.kwargs["attack"]["attack_right"]
                    copy_protected = self.kwargs["attack"]["attack_protected"]
                    # allow full attack
                    self.kwargs["attack"]["attack_right"] = False
                    self.kwargs["attack"]["attack_protected"] = False
                    robust_accuracy_full_attack, dist_dict_full_attack = robustness_evaluation(pl_module, self.data, self.kwargs, return_dist_dict=True)
                    # set to old values
                    self.kwargs["attack"]["attack_right"] = copy_right
                    self.kwargs["attack"]["attack_protected"] = copy_protected
                    robust_accuracy_partial_attack = robust_accuracy
                    dist_dict_partial_attack = dist_dict
                else:
                    robust_accuracy_full_attack = robust_accuracy
                    dist_dict_full_attack = dist_dict
                    robust_accuracy_partial_attack = None
                    dist_dict_partial_attack = None
                    
                pl_module.train()
                # Logging
                pl_module.log(f"Robust-{self.panel_name}/Robust_ACC-{self.kwargs['attack']['attack_type'].upper()}", robust_accuracy_full_attack, on_epoch=True, prog_bar=True)
                if robust_accuracy_partial_attack is not None:
                    pl_module.log(f"Robust-{self.panel_name}/Robust_ACC_Partial_Attack-{self.kwargs['attack']['attack_type'].upper()}", robust_accuracy_partial_attack, on_epoch=True, prog_bar=True)
                # Log perturbation magnitudes
                for key, value in dist_dict_full_attack.items():
                    pl_module.log(f'Robust-{self.panel_name}/{key}', value)
                if dist_dict_partial_attack is not None:
                    for key, value in dist_dict_partial_attack.items():
                        pl_module.log(f"Robust-{self.panel_name}/{key}_Partial_Attack-{self.kwargs['attack']['attack_type'].upper()}", value)
                
            else:
                # skip logging
                return


def box_mask(masks,hm=1/4,wm=1/4):
    N,C,H,W=masks.shape
    mask_protected=torch.ones_like(masks)
    vws=0#W//2+1
    sh=int(H*hm)
    sw=int(W*wm)
    for i in range(N):
        vh = random.randint(0, H-sh)                    
        vw = random.randint(vws, W-sw)
        for j in range(C):
            #vh = random.randint(0, H)                    
            #vw = random.randint(vws, W)
            #h=random.randint(0, H-vh)
            #w=random.randint(0, W-vw)
            
            #print(i,j,vh,sh,vw,sw)
            mask_protected[i,j,vh:vh+sh,vw:vw+sw]=0
    return mask_protected,1-mask_protected           
    
def robustness_evaluation(pl_module, dataloader, config_dict, return_dist_dict=False):
    device = pl_module.device
    pl_module.eval()
    
    all_predictions = []
    all_targets = []
    dist_dict = {"avg-L0-distance": [], "avg-L2-distance": [], "avg-L_inf-distance": [], "avg_MSE": []}
    
    for batch in dataloader():
        if len(batch) == 3:
            data_aug, data_orig, target = batch
            masks = None
        elif len(batch) == 4:
            data_aug, data_orig, target, masks = batch 
            if config_dict["attack"]["attack_box"]:
                mask_box = box_mask(masks, config_dict["attack"]["box_H_mp"], config_dict["attack"]["box_W_mp"])[0]
                masks = torch.clamp(mask_box + masks, min=0, max=1)
        else:
            raise ValueError(f"Unexpected batch size of {len(batch)} encountered")
        
        data = data_orig.to(device)
        adv_soft_prediction, attacked_data = get_attacked_model_prediction(pl_module, data, target, masks, **config_dict)
        adv_prediction_labels = adv_soft_prediction.argmax(1).detach().cpu().numpy()
        
        all_predictions.extend(adv_prediction_labels)
        all_targets.extend(target.detach().cpu().numpy())
        
        l0, l2, mse, linf = compute_distances_between_images(data, attacked_data.to(device))
        dist_dict["avg-L0-distance"].extend(l0.tolist())
        dist_dict["avg-L2-distance"].extend(l2.tolist())
        dist_dict["avg-L_inf-distance"].extend(linf.tolist())
        dist_dict["avg_MSE"].extend(mse.tolist())

    robust_accuracy = accuracy_score(all_predictions, all_targets) * 100
    
    dist_dict["avg-L0-distance"] = np.mean(dist_dict["avg-L0-distance"])
    dist_dict["avg-L2-distance"] = np.mean(dist_dict["avg-L2-distance"])
    dist_dict["avg-L_inf-distance"] = np.mean(dist_dict["avg-L_inf-distance"])
    dist_dict["avg_MSE"] = np.mean(dist_dict["avg_MSE"])
    
    if return_dist_dict:
        return robust_accuracy, dist_dict
    else:
        return robust_accuracy