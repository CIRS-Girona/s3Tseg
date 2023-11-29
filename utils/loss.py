import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):

    def __init__(self, weights=None, epsilon=1e-6):
        super().__init__()
        self.weights = weights
        self.epsilon = epsilon
    
    def forward(self, output, target):
        target = F.one_hot(target, output.size(1)).permute(0,3,1,2)
        output = F.softmax(output, dim=1)
        inter = torch.sum(output*target, dim=(2,3))
        union = torch.sum(output+target, dim=(2,3))
        dice_coeff = 2 * (inter + self.epsilon) / (union + self.epsilon)
        if self.weights is not None:
            assert len(self.weights) == dice_coeff.size(1), \
                'Length of weight tensor must match the number of classes'
            dice_coeff *= self.weights
        return 1 - torch.mean(dice_coeff)  # mean over classes and batch


class TverskyLoss(nn.Module):

    def __init__(self, beta, epsilon=1e-6):
        super().__init__()
        self.beta = beta
        self.epsilon = epsilon
    
    def forward(self, output, target):
        target = F.one_hot(target, output.size(1)).permute(0,3,1,2)
        output = F.softmax(output, dim=1)
        true_pos = torch.sum(output*target, axis=(2,3))
        false_neg = torch.sum(target*(1-output), axis=(2,3))
        false_pos = torch.sum(output*(1-target), axis=(2,3))
        tversky_coeff = (true_pos + self.epsilon) / (true_pos + self.beta*false_pos + 
                                                     (1-self.beta)*false_neg + self.epsilon)
        return 1 - torch.mean(tversky_coeff)


class FocalTverskyLoss(nn.Module):
    
    def __init__(self, beta, gamma, epsilon=1e-6):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.epsilon = epsilon
    
    def forward(self, output, target):
        target = F.one_hot(target, output.size(1)).permute(0,3,1,2)
        output = F.softmax(output, dim=1)
        true_pos = torch.sum(output*target, axis=(2,3))
        false_neg = torch.sum(target*(1-output), axis=(2,3))
        false_pos = torch.sum(output*(1-target), axis=(2,3))
        tversky_coeff = (true_pos + self.epsilon) / (true_pos + self.beta*false_pos + 
                                                     (1-self.beta)*false_neg + self.epsilon)
        focal_tversky = (1-tversky_coeff)**self.gamma
        return torch.mean(focal_tversky)


class FocalLoss(nn.Module):
    
    def __init__(self, alpha, gamma):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, output, target):
        num_classes = output.size(1)
        assert len(self.alpha) == num_classes, \
            'Length of weight tensor must match the number of classes'
        logp = F.cross_entropy(output, target, self.alpha, reduction='none')
        p = torch.exp(-logp)
        focal_loss = (1-p)**self.gamma*logp
        return torch.mean(focal_loss)


class ComboLoss(nn.Module):
    
    def __init__(self, losses, weights):
        super().__init__()
        assert len(weights) == len(losses), \
            'Length of weight array must match the number of loss functions'
        self.losses = losses
        self.weights = weights
    
    def forward(self, output, target):
        return sum([loss(output, target)*wt for loss, wt in zip(self.losses, self.weights)])
