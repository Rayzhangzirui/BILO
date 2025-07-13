# define problems for PDE
import torch
import numpy as np

from PoissonProblem import PoissonProblem

class PoissonProblem2(PoissonProblem):
    '''
    different form of residual
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def residual(self, nn, x):
        
        x.requires_grad_(True)
        
        u_pred = nn(x, nn.pde_params_dict)
        u_x = torch.autograd.grad(u_pred, x,
            create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(u_pred))[0]
        u_xx = torch.autograd.grad(u_x, x,
            create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(u_x))[0]
        res =  u_xx + u_x - 10

        return res, u_pred

    def u_exact(self, x, param:dict):
        e = 2.718281828459045235360287471352
        return 10 * (e* (x+torch.exp(-x)-1) - x) / (e - 1)

