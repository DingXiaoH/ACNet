# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from bisect import bisect_right
from data.data_factory import num_iters_per_epoch
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
import math

# FIXME ideally this would be achieved with a CombinedLRScheduler,
# separating MultiStepLR with WarmupLR
# but the current LRScheduler design doesn't allow it

class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=1.0 / 3,
        warmup_iters=500,
        warmup_method="linear",
        last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]



class WarmupLinearLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        final_lr,
        final_iters,
        warmup_factor=1.0 / 3,
        warmup_iters=500,
        warmup_method="linear",
        last_epoch=-1,
    ):
        assert final_iters > warmup_iters
        self.final_lr = final_lr
        self.final_iters = final_iters
        self.warmup_factor = warmup_factor
        self.warmup_iters = max(warmup_iters, 0)
        self.warmup_method = warmup_method
        super(WarmupLinearLR, self).__init__(optimizer, last_epoch)

    #   last_epoch == 0:            base_lr * warmup_factor
    #   last_epoch == warmup_iters: base_lr
    #   last_epoch == final_iters:  final_lr

    def get_lr(self):
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
            else:
                raise ValueError(
                    "Only 'constant' or 'linear' warmup_method accepted"
                    "got {}".format(self.warmup_method)
                )
            return [
                base_lr
                * warmup_factor
                for base_lr in self.base_lrs
            ]
        else:
            return [
                base_lr - (base_lr - self.final_lr) * float(self.last_epoch - self.warmup_iters) / (
                            self.final_iters - self.warmup_iters)
                for base_lr in self.base_lrs
            ]


class CosineAnnealingExtendLR(torch.optim.lr_scheduler._LRScheduler):
    r"""Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr and
    :math:`T_{cur}` is the number of epochs since the last restart in SGDR:

    .. math::
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 +
        \cos(\frac{T_{cur}}{T_{max}}\pi))

    When last_epoch=-1, sets initial lr as lr.

    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_. Note that this only
    implements the cosine annealing part of SGDR, and not the restarts.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_max (int): Maximum number of iterations.
        eta_min (float): Minimum learning rate. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.

    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(self, optimizer, T_cosine_max, eta_min=0, last_epoch=-1):
        self.eta_min = eta_min
        self.T_cosine_max = T_cosine_max
        super(CosineAnnealingExtendLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch <= self.T_cosine_max:
            return [self.eta_min + (base_lr - self.eta_min) *
                (1 + math.cos(math.pi * self.last_epoch / self.T_cosine_max)) / 2
                for base_lr in self.base_lrs]
        else:
            return [self.eta_min
                for _ in self.base_lrs]


#   LR scheduler should work according the number of iterations
def get_lr_scheduler(cfg, optimizer):
    it_ep = num_iters_per_epoch(cfg)
    if cfg.linear_final_lr is None and cfg.cosine_minimum is None:
        lr_iter_boundaries = [it_ep * ep for ep in cfg.lr_epoch_boundaries]
        return WarmupMultiStepLR(
            optimizer, lr_iter_boundaries, cfg.lr_decay_factor,
            warmup_factor=cfg.warmup_factor,
            warmup_iters=cfg.warmup_epochs * it_ep,
            warmup_method=cfg.warmup_method, )
    elif cfg.cosine_minimum is None:
        return WarmupLinearLR(optimizer, final_lr=cfg.linear_final_lr,
                              final_iters=cfg.max_epochs * it_ep,
                              warmup_factor=cfg.warmup_factor,
                              warmup_iters=cfg.warmup_epochs * it_ep,
                              warmup_method=cfg.warmup_method,)
    else:
        assert cfg.warmup_epochs == 0
        assert cfg.linear_final_lr is None
        assert cfg.lr_decay_factor is None
        if cfg.lr_epoch_boundaries is None:
            print('use cosine decay, the minimum is ', cfg.cosine_minimum)
            return CosineAnnealingLR(optimizer=optimizer, T_max=cfg.max_epochs * it_ep, eta_min=cfg.cosine_minimum)
        else:
            assert len(cfg.lr_epoch_boundaries) == 1
            assert cfg.cosine_minimum > 0
            print('use extended cosine decay, the minimum is ', cfg.cosine_minimum)
            return CosineAnnealingExtendLR(optimizer=optimizer, T_cosine_max=cfg.lr_epoch_boundaries[0] * it_ep,
                                           eta_min=cfg.cosine_minimum)


