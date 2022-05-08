""" Conjugate Gradient method in PyTorch! """
import torch
from torch.optim.optimizer import Optimizer, required


class ConjugateGradientOptimizer(Optimizer):
    """
    Conjugate Gradient method
    
    Notation:
        d_buffer: update vector
        alpha_buffer: alpha
        beta_buffer: beta
    """

    def __init__(self, params, lr=required, weight_decay=0, beta_update_rule='FR', beta_momentum_coeff=1, mu=2):
        self.epoch = 0
        defaults = dict(lr=lr,
                        weight_decay=weight_decay,
                        mu=mu,
                        beta_update_rule=beta_update_rule,
                        beta_momentum_coeff=beta_momentum_coeff
                    )
        print(f'Initialized Conjugate Gradient as {beta_update_rule} Beta Update Rule Mode')
        print(f'beta_momentum_coeff:  {beta_momentum_coeff}')
        super(ConjugateGradientOptimizer, self).__init__(params, defaults)

    def calculate_beta(self, current_grad, prev_grad, d_update_v, beta_update_rule):
        
        current_grad = current_grad.view(-1)
        prev_grad = prev_grad.view(-1)
        d_update_v = d_update_v.view(-1)

        if beta_update_rule == 'FR':
            beta = 0
            denominator = float(torch.dot(prev_grad, prev_grad))
            if denominator != 0:
                numerator = float(torch.dot(current_grad, current_grad))
                beta = min(float(numerator/denominator), 1/2)

        elif beta_update_rule == 'PRP':
            beta = 0
            denominator = float(torch.dot(prev_grad, prev_grad))
            if denominator != 0:
                numerator = float(torch.dot(current_grad, current_grad - prev_grad))
                beta = min(float(numerator/denominator), 1/2)

        elif beta_update_rule == 'HS':
            beta = 0
            denominator = float(torch.dot(d_update_v, current_grad - prev_grad))
            if denominator != 0:
                numerator = float(torch.dot(current_grad, current_grad - prev_grad))
                beta = min(float(numerator/denominator), 1/2)

        elif beta_update_rule == 'DY':
            beta = 0
            denominator = float(torch.dot(d_update_v, current_grad - prev_grad))
            if denominator != 0:
                numerator = float(torch.dot(current_grad, current_grad))
                beta = min(float(numerator/denominator), 1/2)

        elif beta_update_rule == 'HS_DY':

            beta = 0
            denominator = float(torch.dot(d_update_v, current_grad - prev_grad))
            if denominator != 0:
                beta_hs = min(float(torch.dot(current_grad, current_grad - prev_grad))/denominator, 1/2)
                beta_dy = min(float(torch.dot(current_grad, current_grad))/denominator, 1/2)
                beta = max(0, min(beta_hs, beta_dy))

        elif beta_update_rule == 'FR_PRP':
            beta = 0
            denominator = float(torch.dot(prev_grad, prev_grad))
            if denominator != 0:
                beta_fr = min(float(torch.dot(current_grad, current_grad)) / denominator, 1/2)
                beta_prp = min(float(torch.dot(current_grad, current_grad - prev_grad)) / denominator, 1/2)
                beta = max(0, min(beta_fr, beta_prp))

        elif beta_update_rule == 'HZ':
            mu = 0
            for group in self.param_groups:
                mu = group['mu']

            beta = 0
            denominator = float(torch.dot(d_update_v, current_grad - prev_grad))
            if denominator != 0:
                beta_hs = min(float(torch.dot(current_grad, current_grad - prev_grad)) / denominator, 1/2)
                numerator = float(torch.dot(current_grad - prev_grad, current_grad - prev_grad)) * float(torch.dot(current_grad, d_update_v))
                beta = min(beta_hs - mu * (numerator / (denominator** 2)), 1/2)

        return beta
    
    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
            epoch: current epoch to calculate polynomial LR decay schedule.
                   if None, uses self.epoch and increments it.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            current_grad_list = []
            prev_grad_list = []
            d_buffer_list = []
            beta_list = []

            weight_decay = group['weight_decay']
            lr = group['lr']
            beta_update_rule = group['beta_update_rule']
            beta_momentum_coeff = group['beta_momentum_coeff']

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    current_grad_list.append(p.grad.data)

                    state = self.state[p]
                    
                    if 'd_buffer' not in state:
                        d_buffer_list.append(None)
                    else:
                        d_buffer_list.append(state['d_buffer'])
                    
                    if 'prev_grad' not in state:
                        prev_grad_list.append(None)
                    else:
                        prev_grad_list.append(state['prev_grad'])
                    
            for i, param in enumerate(params_with_grad):
                d_update_v = d_buffer_list[i]

                current_grad = current_grad_list[i]
                prev_grad = prev_grad_list[i]

                if weight_decay != 0:
                    current_grad = current_grad.add(param, alpha=weight_decay)
                
                if prev_grad is None: # init
                    beta = 0    
                else:
                    beta = self.calculate_beta(current_grad, prev_grad, d_update_v, beta_update_rule) * beta_momentum_coeff
                beta_list.append(beta)
    
                if d_update_v is None:
                    d_update_v = torch.clone(current_grad).detach()
                    d_buffer_list[i] = d_update_v
                else:
                    d_update_v.mul_(beta).add_(current_grad, alpha=-1)

                alpha_buffer = lr
                param.data.add_(d_update_v, alpha=alpha_buffer)
            
            for p, d_buffer, beta in zip(params_with_grad, d_buffer_list, beta_list):
                state = self.state[p]
                state['prev_grad'] = p.grad
                state['d_buffer'] = d_buffer * beta - p.grad
        return loss
        