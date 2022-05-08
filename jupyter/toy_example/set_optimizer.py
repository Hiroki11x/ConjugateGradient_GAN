# coding: utf-8
import attr
import torch.optim as optim
from cg_optimizer import ConjugateGradientOptimizer


@attr.s
class OptimizerSetting:
    name = attr.ib()
    lr = attr.ib()
    weight_decay = attr.ib()
    model = attr.ib()

    momentum = attr.ib(default=0.9) # sgd, sgd_nesterov
    eps = attr.ib(default=0.001) # adam, rmsprop (term added to the denominator to improve numerical stability )
    alpha = attr.ib(default=0.99) # rmsprop (smoothing constant)
    beta_1 = attr.ib(default=0.5) #adam
    beta_2 = attr.ib(default=0.999) #adam
    eta = attr.ib(default=0.001) # lars coefficient

    # kfac
    damping = attr.ib(default=0.001) 
    
    # for cgd
    beta_update_rule = attr.ib(default='FR')
    beta_momentum_coeff = attr.ib(default=1)
    mu = attr.ib(2)
    max_epoch = attr.ib(200)


def build_optimizer(setting: OptimizerSetting):
    name = setting.name

    # Standard Optimizer
    if name == 'vanilla_sgd':
        return optim.SGD(setting.model, lr = setting.lr, weight_decay=setting.weight_decay)
    elif name == 'momentum_sgd':
        return optim.SGD(filter(lambda p: p.requires_grad, setting.model), lr = setting.lr, momentum=setting.momentum, weight_decay=setting.weight_decay)
    elif name == 'adam':
        return optim.Adam(setting.model, 
                         lr = setting.lr, betas=(setting.beta_1, setting.beta_2), 
                         eps=setting.eps, 
                         weight_decay=setting.weight_decay, 
                         amsgrad=True)

    elif name == 'cgd':
        return ConjugateGradientOptimizer(params=setting.model, 
                                          lr=setting.lr, 
                                          weight_decay=setting.weight_decay, 
                                          beta_update_rule=setting.beta_update_rule, 
                                          beta_momentum_coeff=setting.beta_momentum_coeff,
                                          mu=setting.mu)

    else:
        raise ValueError(
            'The selected optimizer is not supported for this trainer.')
