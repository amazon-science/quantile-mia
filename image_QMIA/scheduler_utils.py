import torch


def build_scheduler(
    scheduler,
    epochs,
    optimizer,
    step_fraction=0.33,
    mode="max",
    l_steps=None,
    step_gamma=0.1,
    lr=None,
):
    if scheduler is None or scheduler == "":
        lr_scheduler = None
    elif scheduler == "cosine":
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=optimizer.param_groups[0]['lr']*min_factor, last_epoch=- 1, verbose=False)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs
        )
    elif scheduler == "linear":
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=epochs
        )
    elif scheduler == "step":
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=int(epochs * step_fraction), gamma=step_gamma
        )
    elif scheduler == "plateau":
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=mode, patience=5
        )
    elif scheduler == "onecycle":
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, lr, epochs=epochs, steps_per_epoch=1
        )
    elif scheduler == "warmupstep":
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=0.1, end_factor=1.0, total_iters=5
                ),
                torch.optim.lr_scheduler.StepLR(
                    optimizer, step_size=int(epochs * step_fraction), gamma=step_gamma
                ),
            ],
            milestones=[5],
        )

    else:
        raise NotImplementedError
    return lr_scheduler


# from torch.optim import LRScheduler
# import warnings
# class WarmupStepLR(LRScheduler):
#     """Decays the learning rate of each parameter group by gamma every
#     step_size epochs. Notice that such decay can happen simultaneously with
#     other changes to the learning rate from outside this scheduler. When
#     last_epoch=-1, sets initial lr as lr.
#
#     Args:
#         optimizer (Optimizer): Wrapped optimizer.
#         step_size (int): Period of learning rate decay.
#         gamma (float): Multiplicative factor of learning rate decay.
#             Default: 0.1.
#         last_epoch (int): The index of last epoch. Default: -1.
#         verbose (bool): If ``True``, prints a message to stdout for
#             each update. Default: ``False``.
#
#     Example:
#         >>> # xdoctest: +SKIP
#         >>> # Assuming optimizer uses lr = 0.05 for all groups
#         >>> # lr = 0.05     if epoch < 30
#         >>> # lr = 0.005    if 30 <= epoch < 60
#         >>> # lr = 0.0005   if 60 <= epoch < 90
#         >>> # ...
#         >>> scheduler = WarmupStepLR(optimizer, step_size=30, gamma=0.1)
#         >>> for epoch in range(100):
#         >>>     train(...)
#         >>>     validate(...)
#         >>>     scheduler.step()
#     """
#
#     def __init__(self, optimizer, step_size, warmup=5, gamma=0.1, last_epoch=-1, verbose=False):
#         self.warmup=warmup
#         self.step_size = step_size
#         self.gamma = gamma
#         super().__init__(optimizer, last_epoch, verbose)
#
#     def get_lr(self):
#         if not self._get_lr_called_within_step:
#             warnings.warn("To get the last learning rate computed by the scheduler, "
#                           "please use `get_last_lr()`.", UserWarning)
#
#         if (self.last_epoch == 0) or (self.last_epoch % self.step_size != 0):
#             return [group['lr'] for group in self.optimizer.param_groups]
#         return [group['lr'] * self.gamma
#                 for group in self.optimizer.param_groups]
#
#     def _get_closed_form_lr(self):
#         return [base_lr * self.gamma ** (self.last_epoch // self.step_size)
#                 for base_lr in self.base_lrs]
