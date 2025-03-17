import torch
from torch import nn
import warmup_scheduler


class Trainer:
    def __init__(self, params, lr, num_epochs, num_steps, device="cuda", warmup_epoch=0.01):
        """
        Trainer class for optimizing models with learning rate scheduling and gradient scaling.
        """
        min_lr = 8e-6
        self.optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.002)

        base_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=1.0,
            end_factor=min_lr / lr,
            total_iters=int(num_steps * (num_epochs - warmup_epoch))
        )

        self.lr_scheduler = warmup_scheduler.GradualWarmupScheduler(
            self.optimizer,
            multiplier=1.0,
            total_epoch=int(warmup_epoch * num_steps),
            after_scheduler=base_scheduler
        )

        self.scaler = torch.cuda.amp.GradScaler()
        self.params = params
        self.second_phase = False
        self.third_phase = False

    def step(self, loss, params):
        """Perform a training step with gradient scaling and learning rate scheduling."""
        scaler = self.scaler
        optimizer = self.optimizer
        lr_scheduler = self.lr_scheduler

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad = torch.nn.utils.clip_grad_norm_(params, max_norm=1.0, norm_type=2)
        scaler.step(optimizer)
        scaler.update()
        lr_scheduler.step()
        optimizer.zero_grad()

        return grad, optimizer.param_groups[0]['lr']


class LabelSmoothingCrossEntropyLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1, dim=-1):
        """
        Label smoothing loss function to mitigate overconfidence in predictions.
        """
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        """Compute label smoothing cross-entropy loss."""
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)

        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
