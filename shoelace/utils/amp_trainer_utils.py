import torch
from torch import nn
import warmup_scheduler


class Trainer:
    def __init__(self, params, lr, num_epochs, num_steps, warmup_epoch=2):
        min_lr = 1e-5
        self.optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.02)
        base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                    T_max=2 * num_steps,
                                                                    eta_min=min_lr)
        self.lr_scheduler = warmup_scheduler.GradualWarmupScheduler(self.optimizer, multiplier=1.,
                                                                    total_epoch=0.01*num_epochs,
                                                                    after_scheduler=base_scheduler)

        self.scaler = torch.cuda.amp.GradScaler()
        self.params = params


    def step(self, loss, params):
        scaler = self.scaler
        optimizer = self.optimizer

        lr_scheduler = self.lr_scheduler
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad = torch.nn.utils.clip_grad_norm_(params, max_norm=.1, norm_type=2)
        lr_scheduler.step()
        scaler.step(optimizer)
        scaler.update()

        optimizer.zero_grad()
        return grad,optimizer.param_groups[0]['lr']


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
