from torch.optim import lr_scheduler
import math

# ================
# Set LR Scheduler
# Reference https://github.com/christiancosgrove/pytorch-spectral-normalization-gan/blob/12dcf945a6359301d63d1e0da3708cd0f0590b19/main.py#L55
# ================


def build_scheduler(opt, optimizerD, optimizerG):

    if opt.scheduler_type == 'ConstLR':
        scheduler_D = lr_scheduler.LambdaLR(optimizerD, lr_lambda=lambda x: 1.0)
        scheduler_G = lr_scheduler.LambdaLR(optimizerG, lr_lambda=lambda x: 1.0)

    elif opt.scheduler_type == 'ExponentialLR':

        scheduler_D = lr_scheduler.ExponentialLR(optimizerD, gamma=0.99999)
        scheduler_G = lr_scheduler.ExponentialLR(optimizerG, gamma=0.99999)

    elif opt.scheduler_type == 'SqrtLR':

        scheduler_D = lr_scheduler.LambdaLR(optimizerD, lr_lambda = lambda steps: 1/math.sqrt(steps+1))
        scheduler_G = lr_scheduler.LambdaLR(optimizerG, lr_lambda = lambda steps: 1/math.sqrt(steps+1))


    elif opt.scheduler_type == 'TTScaleLR ':

        eta_a = 0.75
        eta_b = 0.75

        scheduler_D = lr_scheduler.LambdaLR(optimizerD, lr_lambda = lambda steps: (steps+1)**(-eta_a))
        scheduler_G = lr_scheduler.LambdaLR(optimizerG, lr_lambda = lambda steps: (steps+1)**(-eta_b))

    return scheduler_D, scheduler_G