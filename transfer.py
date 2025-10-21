# Code for transfer learning to test how well changes to non-salient features can be handled
import os

import hydra
import pytorch_lightning as pl
import torch
# logging
import wandb
from omegaconf import DictConfig
from pytorch_lightning.loggers import WandbLogger

from src.models import (SplitTrain, HBaR,
                        StandardClassificationModel, init_backbone_model)
from src.utils import (get_model_hash, load_datamodule,
                       preprocess_omega_conf_for_wandb, set_seed)
from src.models.split_model import get_hard_assignments

def evaluate(cfg):
    if cfg.seed >= 0:
        set_seed(cfg.seed)
    else:
        print("Warning: Running without setting seed!")
    
    if cfg.dataset.name.lower() not in ["counteranimal", "imagenet9", "imagenet-d"]:
        raise ValueError(f"Dataset {cfg.dataset.name} not supported")
    
    test_datamodule = load_datamodule(cfg, setup=True)
    
    run_name, run_hash, group_name = get_model_hash(cfg, return_hash=True, return_group_name=True)
    cfg.run_name = run_name
    cfg.run_hash = run_hash
    cfg.group_name = group_name
    
    # save omega config
    if cfg.wandb.enabled:
        # unwrap config:
        dict_args = preprocess_omega_conf_for_wandb(cfg)
        # Initialize Wandb logging
        logger = WandbLogger(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=run_name,
            save_dir=cfg.wandb.save_dir,
            dir=cfg.wandb.save_dir,
            tags=[cfg.wandb.tag],
            config=dict_args,
        )
    else:
        logger = False
        
    # load model
    backbone_model = init_backbone_model(cfg)
    if cfg.training_type.lower() == "split_train":
        model = SplitTrain(backbone_model=backbone_model, **cfg)
    elif cfg.training_type.lower() == "hbar":
        model = HBaR(backbone_model=backbone_model, **cfg)
    elif cfg.training_type.lower() == "standard":
        model = StandardClassificationModel(backbone_model=backbone_model, **cfg)
    else:
        raise ValueError(f"training_type: {cfg.training_type} does not exist, should be one of split_train or standard")
    
    model.to(cfg.device)
    model.load_state_dict(torch.load(cfg.model_path, weights_only=True), strict=False)
    if cfg.training_type.lower() == "split_train":
        betas_path = cfg.model_path.replace(".pt", ".betas_per_epoch")
        if os.path.exists(betas_path):
            betas_list = torch.load(betas_path, weights_only=False)
            model.betas_per_epoch = betas_list
            
            last_beta_cs = torch.from_numpy(betas_list[-1]).to(model.device)
            model.beta_cs = last_beta_cs
            model.beta_adv = 1.0 - last_beta_cs
            model.betas = torch.stack([model.beta_cs, model.beta_adv], dim=0)
            # model.split_params_to_device(mode="test")
            model.betas = model.betas.to(model.device)
            model.beta_adv = model.beta_adv.to(model.device)
            model.beta_cs = model.beta_cs.to(model.device)
            
            model.hard_betas = get_hard_assignments(
                model.betas.clone(),
                centers=[model.cs_centers, model.adv_center],
                shared_space_variation=model.shared_space_variation,
                shared_space_idx=[1]
            ).to(model.device)
            
            # NOTE: Lightning hook on_test_start will do this
            # model._initialize_feature_hook_betas()
        else:
            raise FileNotFoundError(f"Could not find betas file at {betas_path}")
    
    
    trainer = pl.Trainer(
        logger=logger,
        accelerator='auto',
        enable_checkpointing=False,
    )
    trainer.test(model, datamodule=test_datamodule)
    
    wandb.finish()

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    evaluate(cfg)

if __name__ == "__main__":
    main()
    