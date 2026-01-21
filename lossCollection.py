#!/usr/bin/env python
'''
this class handle the loss function and computing gradient
net = neural net class
dataset = data set class,
pde = pde class, take net and dataset to compute residual
param = list of parameters to optimize, either network weight or pde parameter
lossCollection compute different loss, in particular 
residual loss: residual of pde 
residual gradient loss: derivative of residual w.r.t. pde parameter
data loss: MSE of data
'''
import torch
from util import mse, set_device, set_seed

class lossCollection:
    # loss, parameter, and optimizer
    def __init__(self, net, pde, loss_weight_dict):
        '''
        '''
        self.net = net
        self.pde = pde
        
        # collection of all loss functions
        self.loss_dict = self.pde.loss_dict
        self.loss_weight = {} # dict of active loss: weight

        for k in loss_weight_dict:
            # if k starts with post_ or prior_, it is negative log likelihood, so weight is treated as sigma
            if k.startswith('post_') or k.startswith('prior_'):
                self.loss_weight[k] = 1.0 / loss_weight_dict[k]**2
            else:
                self.loss_weight[k] = loss_weight_dict[k]
    
        self.weighted_loss_comp = {} # component of each loss, weighted
        self.unweighted_loss_comp = {} # component of each loss, unweighted
        self.wtotal = None # total loss for backprop

    
    def get_wloss_sum_comp(self, list_of_loss: list, yes_grad: bool):
        # for bilevel optimization
        # list_of_loss can be empty list
        weighted_sum = 0.0 if list_of_loss else None
        weighted_loss_comp = {}
        unweighted_loss_comp = {}
        with torch.set_grad_enabled(yes_grad):
            for key in list_of_loss:
                unweighted_loss_comp[key] = self.loss_dict[key](self.net)

                weighted_loss_comp[key] = self.loss_weight[key] * unweighted_loss_comp[key]
                weighted_sum += weighted_loss_comp[key]
        
        return weighted_sum, weighted_loss_comp, unweighted_loss_comp

        
            


class EarlyStopping:
    def __init__(self,  **kwargs):
        self.tolerance = kwargs.get('tolerance', 1e-4)
        self.max_iter = kwargs.get('max_iter', 10000)
        self.patience = kwargs.get('patience', 100)
        self.delta_loss = kwargs.get('delta_loss', 0)
        self.burnin = kwargs.get('burnin',1000 )
        self.monitor_loss = kwargs.get('monitor_loss', True)
        self.best_loss = None
        self.counter_param = 0
        self.counter_loss = 0
        self.epoch = 0

    def __call__(self, epoch, loss):
        self.epoch = epoch
        # convert tensor to float
        
        
        if epoch >= self.max_iter:
            print('\nStop due to max iteration')
            return True

        if epoch < self.burnin:
            return False
        
        # monitor loss, stop if not improving
        if self.monitor_loss:
            loss = loss.item()
        
            if loss < self.tolerance:
                print(f'Stop due to loss {loss} < {self.tolerance}')
                return True

            if self.best_loss is None:
                self.best_loss = loss
            elif loss > self.best_loss - self.delta_loss:
                self.counter_loss += 1
                if self.counter_loss >= self.patience:
                    print(f'Stop due to loss patience for {self.counter_loss} steps, best loss {self.best_loss}')
                    return True
            else:
                self.best_loss = loss
                self.counter_loss = 0
        return False


if __name__ == "__main__":


    import sys
    from Options import *
    from DenseNet import *
    from Problems import *


    optobj = Options()
    optobj.parse_args(*sys.argv[1:])
    
    device = set_device('cuda')
    set_seed(0)
    
    # prob = PoissonProblem(p=1, init_param={'D':1.0}, exact_param={'D':1.0})
    prob = create_pde_problem(**optobj.opts['pde_opts'])

    optobj.opts['nn_opts']['input_dim'] = prob.input_dim
    optobj.opts['nn_opts']['output_dim'] = prob.output_dim

    net = DenseNet(**optobj.opts['nn_opts'],
                output_transform=prob.output_transform, 
                params_dict=prob.init_param).to(device)

    dataset = create_dataset_from_pde(prob, optobj.opts['dataset_opts'], optobj.opts['noise_opts'])
    dataset.to_device(device)

    dataset['u_dat_train'] = dataset['u_exact_dat_train']

    params = list(net.parameters())
    


