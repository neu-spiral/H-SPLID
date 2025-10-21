import torch
import torch.nn as nn

from ..attack import Attack

class PGD(Attack):
    r"""
    PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of steps. (Default: 10)
        random_start (bool): using random initialization of delta. (Default: True)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.PGD(model, eps=8/255, alpha=1/255, steps=10, random_start=True)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, model, eps=8 / 255, alpha=2 / 255, steps=10, random_start=True, attack_right=False, attack_protected=False):
        super().__init__("PGD", model, attack_right,attack_protected)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.supported_mode = ["default", "targeted"]

    def forward(self, images, labels,mask_protected=None):
        r"""
        Overridden.
        """
        scale=1
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        if self.attack_protected:
            mask2=mask_protected
            mask1=1-mask_protected   
            #scale=255
            
        elif self.attack_right:
            mask1=torch.ones_like(images)
            mask1[:,:,:,:32] = 0
            mask2=torch.ones_like(images)
            mask2[:,:,:,32:] = 0
        else:
            mask1=torch.ones_like(images)
            mask2=torch.zeros_like(images)

        images=images/scale
        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        loss = nn.CrossEntropyLoss()
        adv_images = images.clone().detach()
        '''
        if self.attack_right:
            if self.random_start:
                # Starting at a uniformly random point
                adv_images = adv_images + mask1*torch.empty_like(adv_images).uniform_(
                    -self.eps, self.eps
                )
                adv_images = torch.clamp(adv_images, min=0, max=1).detach()
        else:
            if self.random_start:
                # Starting at a uniformly random point
                adv_images = adv_images + torch.empty_like(adv_images).uniform_(
                    -self.eps, self.eps
                )
                adv_images = torch.clamp(adv_images, min=0, max=1).detach()
        '''
        if self.random_start:
                # Starting at a uniformly random point
                adv_images = adv_images + mask1*torch.empty_like(adv_images).uniform_(
                    -self.eps, self.eps
                )
                adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            '''
            if self.attack_right:
                masked=adv_images* mask1 + images* mask2
            else:
                masked=adv_images
            '''
            masked=adv_images* mask1 + images* mask2
            outputs = self.get_logits(masked*scale)
            # Added by me, need to check if this is correct
            if type(outputs) == tuple:
                outputs = outputs[0]
            # Calculate loss
            if self.targeted:
                cost = -loss(outputs, target_labels)
            else:
                cost = loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(
                cost, adv_images, retain_graph=False, create_graph=False
            )[0]

            adv_images = adv_images.detach() + self.alpha * grad.sign()

            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images*scale
    