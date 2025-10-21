from src.models.lenet import LeNet3
from src.models.resnet import resnet18, resnet34, resnet50
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights

import torch.nn as nn

def init_backbone_model(config_dict):
    if config_dict['model'] == 'lenet3':
        model = LeNet3(**config_dict)
    elif 'resnet' in config_dict['model']:
        model = resnet_initializer(config_dict)
    else:
        raise ValueError("Unknown model name or not support [{}]".format(config_dict['model']))

    return model


def resnet_initializer(cfg):
    if cfg["dataset"]["channels"] != 3:
        raise ValueError("Expected number of input channels for ResNet to be exactly 3, but is ", cfg["dataset"]["channels"])
    if cfg["dataset"]["resize"] < 224:
        first_conv = False
        maxpool1 = False
    else:
        first_conv = True
        maxpool1 = True
    
    try:
        cfg["resnet"]["norm_layer"]
    except:
        cfg["resnet"]["norm_layer"] = None
        
    try:
        cfg["resnet"]["weights"]
    except:
        cfg["resnet"]["weights"] = None
        
    # Replace the final classification layer if we need different number of classes
    load_pretrained = cfg["resnet"]["weights"] in ["imagenet"]
    target_num_classes = cfg["dataset"]["n_classes"]
    if load_pretrained:
        pretrained_num_classes = 1000
    else:
        pretrained_num_classes = target_num_classes
        
    resnet_kwargs = {
        "num_classes": pretrained_num_classes,
        "first_conv": first_conv,
        "maxpool1": maxpool1,
        "norm_layer": cfg["resnet"]["norm_layer"] if "norm_layer" in cfg["resnet"] else None,
    }
    
    if cfg['model'] == 'resnet18':
        if cfg["resnet"]["weights"] == "imagenet":
            resnet_kwargs["weights"] = ResNet18_Weights.DEFAULT
        model = resnet18(**resnet_kwargs)
    elif cfg['model'] == 'resnet34':
        if cfg["resnet"]["weights"] == "imagenet":
            resnet_kwargs["weights"] = ResNet34_Weights.DEFAULT
        model = resnet34(**resnet_kwargs)
    elif cfg['model'] == 'resnet50':
        if cfg["resnet"]["weights"] == "imagenet":
            resnet_kwargs["weights"] = ResNet50_Weights.DEFAULT
        model = resnet50(**resnet_kwargs)
    
    if load_pretrained and target_num_classes != pretrained_num_classes:
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, target_num_classes)
        print(f"Replaced final classification layer:")
        print(f"  - Original: {in_features} -> {pretrained_num_classes} classes")
        print(f"  - New: {in_features} -> {target_num_classes} classes")

    return model