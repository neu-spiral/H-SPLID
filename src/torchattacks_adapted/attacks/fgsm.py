import torch
import torch.nn as nn

from ..attack import Attack


class FGSM(Attack):
    r"""
    FGSM in the paper 'Explaining and harnessing adversarial examples'
    [https://arxiv.org/abs/1412.6572]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.FGSM(model, eps=8/255)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, model, eps=8 / 255, attack_right=False, attack_protected=False):
        super().__init__("FGSM", model,  attack_right, attack_protected)
        self.eps = eps
        self.supported_mode = ["default", "targeted"]

    def forward(self, images, labels,mask_protected=None):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        if self.attack_protected:
            mask2=mask_protected
            mask1=1-mask_protected   
        elif self.attack_right:
            mask1=torch.ones_like(images)
            mask1[:,:,:,:32] = 0
            mask2=torch.ones_like(images)
            mask2[:,:,:,32:] = 0
        else:
            mask1=torch.ones_like(images)
            mask2=torch.zeros_like(images)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        loss = nn.CrossEntropyLoss()

        
        temp_images = images.clone().detach()
        temp_images.requires_grad = True

        #print("RRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR")
        
        if self.attack_right or self.attack_protected:
            masked=temp_images* mask1 + images* mask2
        else:
            masked=temp_images
        
        outputs = self.get_logits(masked)
        # TT
        if type(outputs) == tuple:
            outputs = outputs[0]

        # Calculate loss
        if self.targeted:
            cost = -loss(outputs, target_labels)
        else:
            cost = loss(outputs, labels)

        # Update adversarial images
        grad = torch.autograd.grad(
            cost, temp_images, retain_graph=False, create_graph=False
        )[0]

        adv_images = temp_images + self.eps * grad.sign()
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        return adv_images
