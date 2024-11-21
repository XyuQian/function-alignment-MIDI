import torch
from torch import nn
import warmup_scheduler


class Trainer:
    def __init__(self, params, lr, num_epochs, num_steps, device="cuda", warmup_epoch=0.01):
        min_lr = 5e-6
        self.optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.002)
        # base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
        #                                                             # T_max=(num_epochs - warmup_epoch) * num_steps,
        #                                                             T_max=int(num_steps*2),
        #                                                             eta_min=min_lr)

        base_scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer,
                                                           start_factor=1.0,  # starting from the initial learning rate
                                                           end_factor=min_lr / lr,  # reaching the minimum LR
                                                           total_iters=int(num_steps * (num_epochs - warmup_epoch)))


        self.lr_scheduler = warmup_scheduler.GradualWarmupScheduler(self.optimizer, multiplier=1.,
                                                                    total_epoch=int(warmup_epoch * num_steps),
                                                                    after_scheduler=base_scheduler)

        self.scaler = torch.cuda.amp.GradScaler()
        self.params = params
        self.second_phase = False
        self.third_phase = False

    def step(self, loss, params):
        scaler = self.scaler
        optimizer = self.optimizer

        lr_scheduler = self.lr_scheduler
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad = torch.nn.utils.clip_grad_norm_(params, max_norm=5., norm_type=2)
        scaler.step(optimizer)
        scaler.update()
        lr_scheduler.step()
        optimizer.zero_grad()
        return grad, optimizer.param_groups[0]['lr']


class LabelSmoothingCrossEntropyLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1, dim=-1):
        super(LabelSmoothingCrossEntropyLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
