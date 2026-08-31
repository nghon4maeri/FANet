import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)

        return 1 - dice

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = (0.5 * BCE) + (0.5 * dice_loss)

        return Dice_BCE


class NegativeAreaDiceBCELoss(nn.Module):
    """DiceBCE with an extra false-positive (negative-area) penalty term.

    Reference: "An Improved Dice Loss for Pneumothorax Segmentation by Mining
    the Information of Negative Areas" (IEEE Access 2020).
    Penalizes predicted-foreground that overlaps ground-truth background:

        fp_penalty = 1 - ( (1-targets)*(1-inputs) ) / ( (1-targets) + 1e-5 )
        loss = DiceBCE + fp_weight * fp_penalty

    In the reference the negative area is derived from a two-stage prediction;
    here we use a soft surrogate on (1 - targets) with the same spirit:
    high penalty when the model is confident foreground where GT is background.

    Usage (fallback F1 when T_B1 is rejected):
        loss_fn = NegativeAreaDiceBCELoss(fp_weight=0.5)
    """
    def __init__(self, fp_weight=0.5, smooth=1.0):
        super(NegativeAreaDiceBCELoss, self).__init__()
        self.fp_weight = fp_weight
        self.smooth = smooth

    def forward(self, inputs, targets, smooth=None):
        if smooth is None:
            smooth = self.smooth
        inputs = torch.sigmoid(inputs).view(-1)
        targets = targets.view(-1)

        # standard Dice
        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        dice_bce = 0.5 * BCE + 0.5 * dice_loss

        # negative-area (FP) penalty: foreground predicted on GT-background
        bg_targets = (1.0 - targets)
        # soft overlap of predicted fg with GT bg, normalized by GT bg
        fp_overlap = (inputs * bg_targets).sum()
        fp_norm = bg_targets.sum() + smooth
        fp_penalty = 1.0 - (fp_overlap + smooth) / (inputs.sum() + smooth + 1e-8)
        # alternative bounded form: how much of predicted fg is on bg
        fp_frac = (inputs * bg_targets).sum() / (inputs.sum() + smooth + 1e-8)

        return dice_bce + self.fp_weight * fp_frac


class FocalDiceBCELoss(nn.Module):
    """(Reserved) Not used in Phase 4 primary cells."""
    pass
