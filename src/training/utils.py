import torch
from torch.optim.lr_scheduler import _LRScheduler
import src.torchattacks_adapted as torchattacks
from src.training.corruption_attack import CorruptionAttack


def get_attack_modality(backbone_model, **kwargs):
    
    if kwargs['attack']['attack_type'] == 'pgd':
        attack = torchattacks.PGD(backbone_model, eps=kwargs['attack']['epsilon'], 
                                  alpha=kwargs['attack']['pgd_alpha'], 
                                  steps=kwargs['attack']['pgd_steps'], 
                                  attack_right=kwargs['attack']["attack_right"],
                                  attack_protected=kwargs['attack']["attack_protected"])
    elif kwargs['attack']['attack_type'] == 'fgsm':
        attack = torchattacks.FGSM(backbone_model, eps=kwargs['attack']['epsilon'], 
                                   attack_right=kwargs['attack']["attack_right"],
                                   attack_protected=kwargs['attack']["attack_protected"])
    elif kwargs['attack']['attack_type'] == 'autoattack':
        attack = torchattacks.AutoAttack(backbone_model, eps=kwargs['attack']['epsilon'], 
                                        norm=kwargs['attack']['norm'], n_classes=kwargs['attack']['n_classes'],
                                        version=kwargs['attack']['version'], seed=kwargs['attack']["seed"], 
                                        attack_right=kwargs['attack']["attack_right"],
                                        attack_protected=kwargs['attack']["attack_protected"])
    elif kwargs['attack']['attack_type'] == 'corruption':
        attack = CorruptionAttack(backbone_model, 
                                 corruption_type=kwargs['attack']['corruption_type'],
                                 severity=kwargs['attack']['severity'],
                                 attack_right=kwargs['attack']["attack_right"],
                                 attack_protected=kwargs['attack']["attack_protected"])
    else:
        raise ValueError(f"Unknown attack type: {kwargs['attack']['attack_type']}")
    
    return attack

def compute_distances_between_images(a, b):
    # description taken from: https://github.com/ndb796/PyTorch-Adversarial-Attack-Baselines-for-ImageNet-CIFAR10-MNIST
    # Intepretation assumes: Generally, each pixel value is normalized between [0, 1].
    # A perturbation with L0 norm of 1,000 could change 1,000 pixels (the number of changed pixels)
    l0 = torch.norm((a - b).view(a.shape[0], -1), p=0, dim=1)
    # A perturbation with L2 norm of 1.0 could change one pixel by 255, ten pixels by 80, 100 pixels by 25, or 1000 pixels by 8.
    l2 = torch.norm((a - b).view(a.shape[0], -1), p=2, dim=1)
    # A perturbation with Linf norm of 0.003922 could change all pixels by 1 (the maximum changeable amount of each pixel).
    linf = torch.norm((a - b).view(a.shape[0], -1), p=float('inf'), dim=1)
    # A perturbation with MSE of 0.001 or lower generally seems imperceptible to humans.
    mse = (a - b).view(a.shape[0], -1).pow(2).mean(1)
    return l0, l2, mse, linf

def set_optimizer(config_dict, model, train_loader):
    """ bag of tricks set-ups"""
    print("test")

    optimizer_init_lr = config_dict['warmup_lr'] if config_dict['warmup'] else config_dict['learning_rate']

    # Select optimizer based on config
    optimizer_type = config_dict.get('optimizer', 'adam').lower()

    if optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=optimizer_init_lr,
            momentum=config_dict.get('momentum', 0.9),
            weight_decay=config_dict.get("weight_decay", 0.0)
        )
    elif optimizer_type == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            optimizer_init_lr,
            weight_decay=config_dict["weight_decay"]
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_type}. Supported optimizers: 'sgd', 'adam'")

    scheduler = None
    if config_dict['lr_scheduler'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config_dict['epochs'] * len(train_loader), eta_min=4e-08)
    elif config_dict['lr_scheduler'] == "hbar_default":
        """Set the learning rate of each parameter group to the initial lr decayed
                by gamma once the number of epoch reaches one of the milestones
        """
        if config_dict['data_code'] == 'cmnist':
            epoch_milestones = [65, 90]
        else:
            raise NotImplementedError(f"LR Scheduler for dataset {config_dict['data_code']} not defined")
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[i * len(train_loader) for i in epoch_milestones], gamma=0.5)
    else:
        scheduler = None
    if config_dict['warmup']:
        scheduler = GradualWarmupScheduler(optimizer, multiplier=config_dict['learning_rate']/config_dict['warmup_lr'], total_iter=config_dict['warmup_epochs'] * len(train_loader), after_scheduler=scheduler)
        
    return optimizer, scheduler

class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier
        total_iter: target learning rate is reached at total_iter, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_iter, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier <= 1.:
            raise ValueError('multiplier should be greater than 1.')
        self.total_iter = total_iter
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_iter:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_iter + 1.) for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if self.finished and self.after_scheduler:
            return self.after_scheduler.step(epoch)
        else:
            return super(GradualWarmupScheduler, self).step(epoch)