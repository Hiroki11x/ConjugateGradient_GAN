from optimizers import *
import torch.optim as optim
import torch

import sys
import os
from pytorch_dnn_arsenal.optimizer import build_optimizer, OptimizerSetting

def set_optimizers(optimizer, model, lr, momentum, beta1, beta2, eps, beta_momentum_coeff):

    if optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=(beta1, beta2), eps=eps)
    elif optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    elif optimizer == 'momentum_sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    elif optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)

    elif optimizer == 'cgd_fr':
        optimizer = build_optimizer(
            OptimizerSetting(name='cgd',
                            weight_decay = 0,
                            lr=lr,
                            beta_update_rule='FR',
                            beta_momentum_coeff = beta_momentum_coeff,
                            model=model))
    
    elif optimizer == 'cgd_prp':
        optimizer = build_optimizer(
            OptimizerSetting(name='cgd',
                            weight_decay = 0,
                            lr=lr,
                            beta_update_rule='PRP',
                            beta_momentum_coeff = beta_momentum_coeff,
                            model=model)) 

    elif optimizer == 'cgd_hs':
        optimizer = build_optimizer(
            OptimizerSetting(name='cgd',
                            weight_decay = 0,
                            lr=lr,
                            beta_update_rule='HS',
                            beta_momentum_coeff = beta_momentum_coeff,
                            model=model))

    elif optimizer == 'cgd_dy':
        optimizer = build_optimizer(
            OptimizerSetting(name='cgd',
                            weight_decay = 0,
                            lr=lr,
                            beta_update_rule='DY',
                            beta_momentum_coeff = beta_momentum_coeff,
                            model=model))

    elif optimizer == 'cgd_hs_dy':
        optimizer = build_optimizer(
            OptimizerSetting(name='cgd',
                            weight_decay = 0,
                            lr=lr,
                            beta_update_rule='HS_DY',
                            beta_momentum_coeff = beta_momentum_coeff,
                            model=model))

    elif optimizer == 'cgd_fr_prp':
        optimizer = build_optimizer(
            OptimizerSetting(name='cgd',
                            weight_decay = 0,
                            lr=lr,
                            beta_update_rule='FR_PRP',
                            beta_momentum_coeff = beta_momentum_coeff,
                            model=model))

    elif optimizer == 'cgd_hz':
        optimizer = build_optimizer(
            OptimizerSetting(name='cgd',
                            weight_decay = 0,
                            lr=lr,
                            beta_update_rule='HZ',
                            beta_momentum_coeff = beta_momentum_coeff,
                            model=model))

    return optimizer
