# train.py
import os

import hydra
import pytorch_lightning as pl
# logging
import wandb
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from src.eval import RobustnessMetricsCallback
from src.models import (HBaR, SplitTrain,
                        StandardClassificationModel, init_backbone_model)
from src.utils import (get_model_hash, load_datamodule, load_model_from_config,
                       overwrite_split_hps_with_zero,
                       preprocess_omega_conf_for_wandb, save_model,
                       save_omega_config, set_seed)

def train(cfg):
    if cfg.seed >= 0:
        set_seed(cfg.seed)
    else:
        print("Warning: Running without setting seed!")

    datamodule = load_datamodule(cfg, setup=False)
    
    # overwrite split_hps to zero
    if cfg.training_type != "split_train" and 'split_hps' in cfg:
        cfg = overwrite_split_hps_with_zero(cfg)
    if cfg.attack.attack_protected and not ("coco" in cfg.dataset.name.lower() or "isic2017" in cfg.dataset.name.lower()):
        print("WARNING: attack_protected is set to True, but only works for COCO and ISIC 2017 datasets. Will be set to False")
        cfg.attack.attack_protected = False
    save_omega_config(cfg)
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
        

    if cfg.pretrained_model_path is None:
        # Load backbone model, e.g, LeNet3 with generic model loader function
        backbone_model = init_backbone_model(cfg)
    else:
        backbone_model = load_model_from_config(cfg.pretrained_model_path)
    
    datamodule.setup()
         
    if cfg.training_type.lower() == "split_train":
        model = SplitTrain(backbone_model=backbone_model, **cfg)
    elif cfg.training_type.lower() == "hbar":
        model = HBaR(backbone_model=backbone_model, **cfg)
    elif cfg.training_type.lower() == "standard":
        model = StandardClassificationModel(backbone_model=backbone_model, **cfg)
    else:
        raise ValueError(f"training_type: {cfg.training_type} does not exist, should be one of split_train or standard")
    
    metrics_cb = RobustnessMetricsCallback(datamodule.val_dataloader, panel_name="val", **cfg)
    ckpt_conf = getattr(cfg, "checkpoint", None)
    if ckpt_conf is not None and getattr(ckpt_conf, "enabled", False):
        ckpt_dir = ckpt_conf.ckpt_dir + f"/{cfg.run_hash}"
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
            
        save_omega_config(cfg, ckpt_dir + f"/{cfg.run_hash}.yaml")
        checkpoint_best = ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="{epoch}"+f"-{cfg.run_name}-best",
            monitor=ckpt_conf.monitor or "val/ACC",
            mode=ckpt_conf.mode or "max",
            save_top_k=1,
            save_last=True,
            save_on_train_epoch_end=True,
        )
        callbacks = [checkpoint_best, metrics_cb]
        enable_ckpt = True
    else:
        callbacks = [metrics_cb]
        enable_ckpt = False
        
    trainer = pl.Trainer(
        max_epochs=cfg.epochs,
        logger=logger,
        accelerator=cfg.accelerator,
        devices=cfg.devices if cfg.accelerator != "cpu" else None,
        inference_mode=True, 
        enable_checkpointing=enable_ckpt,
        check_val_every_n_epoch=cfg.log_interval,
        callbacks=callbacks,
        )
    
    fit_kwargs = {"model": model, "datamodule": datamodule}
    if getattr(cfg, "resume_from_checkpoint", None):
        fit_kwargs["ckpt_path"] = cfg.resume_from_checkpoint
    trainer.fit(**fit_kwargs)
    
    trainer.test(model, dataloaders=datamodule.test_dataloader())
    save_model(model, cfg)
        
    # Store the run id, before wandb finishes
    run_id = wandb.run.id if wandb.run else None
    
    if wandb.run:
        wandb.finish()
    
    # Return the run id for the wandb run
    return run_id

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    main()