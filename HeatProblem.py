#!/usr/bin/env python
# Heat equation, infer initial condition
import torch
from torch import nn
import os
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from util import generate_grf, add_noise,  griddata_subsample

from MatDataset import MatDataset
from DenseNet import DenseNet, ParamFunction

# the setup is similar to BurgerProblem, BiLO has the form u(x,t, u0(x)).
from BaseProblem import BaseProblem
from BurgerProblem import BurgerProblem, BurgerDenseNet

  
class HeatProblem(BurgerProblem):
    def __init__(self, **kwargs):
        BaseProblem.__init__(self)
        self.input_dim = 2
        self.output_dim = 1
        self.opts=kwargs
 
        self.testcase = kwargs['testcase']
        self.use_exact_u0 = kwargs['use_exact_u0']
        
        self.param = {'u0': 1.0}

        self.dataset = MatDataset(kwargs['datafile'])
        self.D = self.dataset['D']
        self.loss_dict['l2grad'] = self.get_l2grad
    
    def residual(self, nn, X_in):
        
        X_in.requires_grad_(True)

        t = X_in[:, 0:1]
        x = X_in[:, 1:2]
        
        # Concatenate sliced tensors to form the input for the network
        X = torch.cat((t,x), dim=1)

        u = nn(X, nn.pde_params_dict)
        
        u_t = torch.autograd.grad(u, t,
            create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(u))[0]
        u_x = torch.autograd.grad(u, x,
            create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(u))[0]
        u_xx = torch.autograd.grad(u_x, x,
            create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(u_x))[0]
        
        res = u_t - self.D * u_xx

        return res, u

if __name__ == "__main__":
    import sys
    from Options import *
    from DenseNet import *
    from Problems import *


    optobj = Options()
    optobj.opts['pde_opts']['problem'] = 'heat'
    optobj.opts['pde_opts']['trainable_param'] = 'u0'


    optobj.parse_args(*sys.argv[1:])
    
    
    device = set_device('cuda')
    set_seed(0)
    
    print(optobj.opts)

    prob = HeatProblem(**optobj.opts['pde_opts'])
    pdenet = prob.setup_network(**optobj.opts['nn_opts'])
    prob.setup_dataset(optobj.opts['dataset_opts'], optobj.opts['noise_opts'])

    prob.make_prediction(pdenet)
    prob.visualize(savedir=optobj.opts['logger_opts']['save_dir'])


