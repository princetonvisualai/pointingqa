import math
from torch.optim.lr_scheduler import LambdaLR

class WarmupISRScheduler(LambdaLR):
    """ Inverse Square Root learning rate schedule used in T5
    """
    def __init__(self, optimizer, warmup_steps, warmup_from_zero=False, **kwargs):
        self.warmup_steps = warmup_steps
        self.warmup_from_zero = warmup_from_zero
        super(WarmupISRScheduler, self).__init__(optimizer, self.lr_lambda, last_epoch=-1)

    def lr_lambda(self, step):
        if self.warmup_from_zero:
            return math.sqrt(self.warmup_steps) * (float(min(step, self.warmup_steps) / self.warmup_steps) / float(math.sqrt(max(step, self.warmup_steps))))
        return float(1) / float(math.sqrt(max(step, self.warmup_steps)))

    def get_last_lr(self):
        return super().get_last_lr()
