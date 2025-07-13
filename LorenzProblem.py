# define problems for PDE
import torch
from MatDataset import MatDataset


import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from util import generate_grf

from FKproblem import FKproblem
from GBMproblem import GBMproblem
from PoissonProblem import PoissonProblem
from SimpleODEProblem import SimpleODEProblem


class LorenzProblem():
    def __init__(self, **kwargs):
        super().__init__()
        self.input_dim = 1
        self.output_dim = 3
        
        self.init_param = {'sigma':1.0, 'rho':1.0, 'beta':1.0}
        self.exact_param = {'sigma':10.0, 'rho':15.0, 'beta':8.0/3.0}
        u0 = torch.tensor([-8.0,  7.0, 27.0])

        self.output_transform = torch.nn.Module()
        self.output_transform.register_buffer('u0', u0)
        self.output_transform.forward = lambda x, u: self.output_transform.u0 + u*x
        

    def residual(self, nn, x, param:dict):
        ### much slower than method2
        # u_pred = nn(x)
        # u_t = torch.autograd.functional.jacobian(lambda t: nn(t), x, create_graph=True)
    
        # ## sum over last 2 dimensions
        # u_t = u_t.sum(dim=(2,3))
        # # lorenz system residual
        # res = torch.zeros_like(u_pred)
        # res[:,0] = u_t[:,0] - (param['sigma'] * (u_pred[:,1] - u_pred[:,0]))
        # res[:,1] = u_t[:,1] - (u_pred[:,0] * (param['rho'] - u_pred[:,2]) - u_pred[:,1])
        # res[:,2] = u_t[:,2] - (u_pred[:,0] * u_pred[:,1] - param['beta'] * u_pred[:,2])

        ####method2
        u_pred = nn(x)  # Assuming x.shape is (batch, 1)

        # Initialize tensors
        u_t = torch.zeros_like(u_pred)
        res = torch.zeros_like(u_pred)

        # Compute gradients for each output dimension and adjust dimensions
        for i in range(u_pred.shape[1]):
            grad_outputs = torch.ones_like(u_pred[:, i])
            u_t_i = torch.autograd.grad(u_pred[:, i], x, grad_outputs=grad_outputs, create_graph=True, retain_graph=True)[0]
            u_t[:, i] = u_t_i[:, 0]  # Adjust dimensions

        # Perform your operations
        res[:, 0] = u_t[:, 0] - (param['sigma'] * (u_pred[:, 1] - u_pred[:, 0]))
        res[:, 1] = u_t[:, 1] - (u_pred[:, 0] * (param['rho'] - u_pred[:, 2]) - u_pred[:, 1])
        res[:, 2] = u_t[:, 2] - (u_pred[:, 0] * u_pred[:, 1] - param['beta'] * u_pred[:, 2])

        return res, u_pred

    def f(self, x):
        pass

    def u_exact(self, x, param:dict):
        pass
