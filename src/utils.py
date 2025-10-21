import hashlib
import json
import os
import random
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import yaml
from omegaconf import OmegaConf

from src.datasets.datamodule import ColorDataModule, GrayScaleDataModule
from src.models import (SplitTrain, HBaR,
                        StandardClassificationModel, init_backbone_model)


def unpack_conf(d, key):
    for subkey, value in d[key].items():
        d[f"{key}.{subkey}"] = value
    d.pop(key)
    return d

def zero_out_conf(d, key):
    for subkey, _ in d[key].items():
        d[f"{key}.{subkey}"] = 0
    d.pop(key)
    return d

def overwrite_split_hps_with_zero(cfg):
    for subkey, _ in cfg["split_hps"].items():
        cfg["split_hps"][subkey] = 0
    return cfg

def preprocess_omega_conf_for_wandb(cfg):
    dict_args = deepcopy(dict(OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)))
    # unpack nested dictionaries for proper logging
    for name in ["wandb", "attack", "dataset"]:
        dict_args = unpack_conf(dict_args, name)
    
    if cfg.training_type == "split_train":
        dict_args = unpack_conf(dict_args, "split_hps")
    else:
        if "split_hps" in dict_args:
            # set all split_hps to zero for proper logging
            dict_args = zero_out_conf(dict_args, "split_hps")
    # create boolean flag to track if pretraining was used
    if cfg.pretrained_model_path is None:
        dict_args["pretraining"] = False
    else:
        dict_args["pretraining"] = True
    return dict_args


def save_omega_config(cfg, save_path=None):
    # adapted from https://www.geeksforgeeks.org/pyyaml-dump-format-python/
    dict_args = deepcopy(OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))
    if save_path is None:
        model_path = get_model_path(cfg)
        config_path = str(model_path).replace(".pt", ".yaml")
    else:
        config_path = save_path
    # make directories if they do not exist
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as file:
        yaml.dump(dict_args, file, default_flow_style=False, indent=4)


def get_group_hash(cfg) -> str:
    cfg_dict = deepcopy(dict(OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)))
    cfg_dict.pop("seed", None)
    config_str = json.dumps(cfg_dict, sort_keys=True)
    group_hash = hashlib.md5(config_str.encode("utf-8")).hexdigest()
    return group_hash

def get_run_hash(cfg):
    converted_cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    config_str = json.dumps(converted_cfg, sort_keys=True)
    run_hash = hashlib.md5(config_str.encode()).hexdigest()
    return run_hash

def get_model_hash(cfg, return_hash=False, return_group_name=False) -> str:
    """Generate a unique model name for model configuration"""
    if cfg.run_name is None or cfg.run_hash is None:
        if 'split_hps' in cfg:
            run_name = f"{cfg.dataset.name}-{cfg.training_type}-{cfg.model}-lx-{cfg.split_hps.lambda_x}-lyc-{cfg.split_hps.lambda_y_cluster}-ly-{cfg.split_hps.lambda_y}-lxs-{cfg.split_hps.lambda_x_shared}-lc-{cfg.split_hps.lambda_cluster}-ls-{cfg.split_hps.lambda_shared}-bs-{cfg.split_hps.beta_step}-{cfg.seed}"
        else:
            run_name = f"{cfg.dataset.name}-{cfg.training_type}-{cfg.model}-bs-{cfg.batch_size}-{cfg.seed}"
        run_hash = get_run_hash(cfg)
        run_name = run_name + "-" + run_hash
    else:
        run_name = cfg.run_name
        run_hash = cfg.run_hash
    
    if 'split_hps' in cfg:
        group_hash =  f"lx-{cfg.split_hps.lambda_x}-lyc-{cfg.split_hps.lambda_y_cluster}-ly-{cfg.split_hps.lambda_y}-lxs-{cfg.split_hps.lambda_x_shared}-lc-{cfg.split_hps.lambda_cluster}-ls-{cfg.split_hps.lambda_shared}-bs-{cfg.split_hps.beta_step}-{get_group_hash(cfg)}"
    else:
        group_hash = f"lr-{cfg.learning_rate}-wd-{cfg.weight_decay}-bs-{cfg.batch_size}-epochs-{cfg.epochs}-{get_group_hash(cfg)}"
    if return_hash:
        if return_group_name:
            return run_name, run_hash, group_hash
        else:
            return run_name, run_hash
    else:
        if return_group_name:
            return run_name, group_hash
        return run_name

def get_model_path(cfg) -> Path:
    """Get path for saving/loading pretrained model"""
    model_hash = get_model_hash(cfg)
    if cfg.model_dir is None:
        model_dir = Path('./assets/models')
    else:
        model_dir = Path(cfg.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir / f"{model_hash}.pt"

def save_model(model: torch.nn.Module, cfg) -> None:
    """Save pretrained model with its config hash"""
    model_path = get_model_path(cfg)
    torch.save(model.state_dict(), model_path)
    print(f"Saved pretrained model to {model_path}")
    if cfg.training_type == "split_train":
        torch.save(model.betas_per_epoch, str(model_path).replace(".pt", ".betas_per_epoch"))

def load_model(model: torch.nn.Module, cfg) -> tuple[bool, torch.nn.Module]:
    """Load pretrained model if exists with matching config"""
    model_path = get_model_path(cfg)
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, weights_only=True))
        print(f"Loaded pretrained model from {model_path}")
        return True, model
    return False, model

def load_model_explicit(pretrained_model_path, datamodule=None, return_cfg=False):
    """Load pretrained model if exists with matching config"""
    model_cfg_path = str(pretrained_model_path).replace(".pt", ".yaml")
    cfg = OmegaConf.load(model_cfg_path)     
    backbone_model = load_model_from_config(pretrained_model_path)

    if cfg.training_type.lower() == "split_train":
        if "beta_update_step_fraction" not in cfg.split_hps:
            cfg.split_hps["beta_update_step_fraction"] = 1.0
        if "shared_space_variation" not in cfg.split_hps:
            cfg.split_hps["shared_space_variation"] = 0.025
        if "beta_init_sample_fraction" not in cfg.split_hps:
            cfg.split_hps["beta_init_sample_fraction"] = 1.0
        model = SplitTrain(backbone_model=backbone_model, **cfg)
        model.to(cfg.device)

        # init beta split values and centers
        if datamodule is None:
            raise ValueError("Datamodule cannot be none if split_train is used")
        else:
            model.init_split_module(datamodule.train_dataloader())
    elif cfg.training_type.lower() == "hbar":
        model = HBaR(backbone_model=backbone_model, **cfg)
        model.to(cfg.device)
    elif cfg.training_type.lower() == "standard":
        model = StandardClassificationModel(backbone_model=backbone_model, **cfg)
        model.to(cfg.device)
    else:
        raise ValueError(f"training_type: {cfg.training_type} does not exist, should be one of split_train or standard")
    # if cfg.training_type.lower() == "split_train":
    #     model.split_params_to_device()
    if return_cfg:
        return model, cfg
    else:
        return model



def load_model_from_config(pretrained_model_path, return_cfg=False):
    model_cfg_path = str(pretrained_model_path).replace(".pt", ".yaml")
    model_cfg = OmegaConf.load(model_cfg_path)
    backbone_model = init_backbone_model(model_cfg)
    state_dict = torch.load(pretrained_model_path, map_location="cpu")
    
    # Handle both with and without "model." prefix for backwards compatibility
    new_dict = OrderedDict()
    has_model_prefix = any(key.startswith("model.") for key in state_dict.keys())
    
    if has_model_prefix:
        # Old format: remove "model." prefix
        for key, value in state_dict.items():
            if key.startswith("model."):
                new_dict[key.replace("model.", "")] = value
            else:
                new_dict[key] = value
    else:
        # New format: use directly
        new_dict = state_dict
    
    try:
        backbone_model.load_state_dict(new_dict, strict=False)
    except RuntimeError as e:
        raise RuntimeError(f"Model loading failed - config/checkpoint mismatch: {e}")
    
    if return_cfg:
        return backbone_model, model_cfg
    else:
        return backbone_model
    
def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

def load_datamodule(cfg, setup=False):
    torch.set_float32_matmul_precision('medium')
    if cfg.dataset.channels == 1:
        datamodule = GrayScaleDataModule(
            cfg.dataset.name,
            batch_size=cfg.batch_size,
            val_split_ratio=cfg.val_split_ratio,
            resize=cfg.dataset.resize,
            data_dir=cfg.dataset.data_dir,
            num_workers=cfg.num_workers,
            channel_means=torch.tensor(cfg.dataset.channel_means),
            channel_stds=torch.tensor(cfg.dataset.channel_stds),
        )
    elif cfg.dataset.channels == 3:
        datamodule = ColorDataModule(
            cfg.dataset.name,
            batch_size=cfg.batch_size,
            val_split_ratio=cfg.val_split_ratio,
            resize=cfg.dataset.resize,
            data_dir=cfg.dataset.data_dir,
            num_workers=cfg.num_workers,
            channel_means=torch.tensor(cfg.dataset.channel_means),
            channel_stds=torch.tensor(cfg.dataset.channel_stds),
            subset_path=cfg.dataset.subset_path if 'subset_path' in cfg.dataset else None,
            sample_fraction_per_class=cfg.dataset.sample_fraction_per_class if 'sample_fraction_per_class' in cfg.dataset else None,
            evaluate_mode=cfg.dataset.evaluate_mode if 'evaluate_mode' in cfg.dataset else None,
        )
    else:
        raise ValueError(f"Unknown number of channels {cfg.dataset.channels}")
    
    if setup:
        datamodule.setup()
        
    return datamodule