import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from typing import List, Tuple



class WeightedCrossEntropyLoss(_Loss):
    '''
    Sampled reweighted Cross Entropy loss
    only accept one demensions target and two demension input.
    '''

    def __init__(self, ):
        super(WeightedCrossEntropyLoss, self).__init__()

    def __call__(self, input:torch.Tensor, target:torch.Tensor, sample_weight):
        probs = F.log_softmax(input, dim = 1)
        #if target.ndimension():
        if target.ndimension() == 1:
            #print(target.shape)
            target = target.expand(1, *target.shape)
            target = target.transpose(1, 0)
        one_hot = torch.zeros_like(probs).scatter_(1, target, 1)
        probs = probs * one_hot * -1.0
        loss = torch.sum(probs, 1)
        loss = loss * sample_weight
        loss = torch.mean(loss)
        return loss

class LabelSmoothCrossEntropyLoss(_Loss):

    def __init__(self, eps = 0.1, class_num = 1000):
        super(LabelSmoothCrossEntropyLoss, self).__init__()

        self.min_value = eps / class_num
        self.eps = eps


    def __call__(self, pred:torch.Tensor, target:torch.Tensor):

        epses = self.min_value * torch.ones_like(pred)
        log_probs = F.log_softmax(pred, dim=1)

        if target.ndimension() == 1:
            #print(target.shape)
            target = target.expand(1, *target.shape)
            #print(target, 'dwa')
            target = target.transpose(1, 0)
        target = torch.zeros_like(log_probs).scatter_(1, target, 1)
        target = target.type(torch.float)
        target = target * (1 - self.eps) + epses

        #print(target, 'fff')
        element_wise_mul = log_probs * target * -1.0

        loss = torch.sum(element_wise_mul, 1)
        loss = torch.mean(loss)

        return loss


class AuxClassifersLoss(_Loss):

    def __init__(self, BasicLoss, weights:List[float]):
        super(AuxClassifersLoss, self).__init__()
        self.BasicLoss = BasicLoss
        self.weights = weights
        #print('AuxCls', self.BasicLoss)

    def __call__(self, preds:List[torch.Tensor], target):

        loss = 0
        for pred in preds:
            loss = loss + self.BasicLoss(pred, target)
        return loss


class GaussianWeightedCELoss(_Loss):

    def __init__(self, sigma = 1.0):
        super(GaussianWeightedCELoss, self).__init__()
        self.sigma = sigma
        self.WCE = WeightedCrossEntropyLoss()

    def __call__(self, input:torch.Tensor, target:torch.Tensor):
        sample_weight = torch.randn((input.size(0), 1)) * self.sigma
        sample_weight = sample_weight.to(input.device)
        sample_weight = sample_weight + torch.ones_like(sample_weight)
        loss = self.WCE(input, target, sample_weight)
        return loss


