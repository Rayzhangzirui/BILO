#!/usr/bin/env python
# define problems for PDE
import torch
import os
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from util import generate_grf, add_noise, error_logging_decorator

from BayesianProblem import BayesianProblem
from MatDataset import MatDataset

from DenseNet import DenseNet
import torch.nn as nn


class CuspNet(DenseNet):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
    
        self.phi_a = None

    def forward(self, x, pde_params_dict=None):
        z = pde_params_dict['z']
        phi_a = torch.abs(x-z)
        # retain grad for phi_a
        phi_a.requires_grad_(True)
        self.phi_a = phi_a

        x_with_z = torch.cat((x, phi_a), dim=1)
        
        if self.modifiedmlp:
            u = self.forward_modified(x_with_z, pde_params_dict)
        else:
            u = self.basic_forward(x_with_z, pde_params_dict)
        
        # ad-hoc
        # make u positive
        # u = torch.exp(u)

        u = self.output_transform(x, u, pde_params_dict)

        # ad-hoc
        # manual scall  
        u = 50*u
        # u = u * self.params_expand['lmbd'] / self.params_expand['mu']

        return u

class ExactSolution(DenseNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.char_param = kwargs['char_param']
    
    def forward(self, x, pde_params_dict:dict):
        true_param = {k: pde_params_dict[k] * self.char_param[k] for k in pde_params_dict}
        u = PDE.u_exact(x, **true_param)
        return u
    


class PDE:
    @staticmethod
    def fv(x, **param):
        mu = param['mu']
        z = param['z']
        L = param['L']

        sqrt_mu = torch.sqrt(mu)
        csch_term = 1 / torch.sinh(sqrt_mu * L)
        sinh_term1 = torch.sinh(sqrt_mu * torch.minimum(x, z))
        sinh_term2 = torch.sinh(sqrt_mu * (L - torch.maximum(x, z)))

        v = (1 / sqrt_mu) * csch_term * sinh_term1 * sinh_term2
        return v

    @staticmethod
    def fu(x, **param):
        return param['lmbd'] * PDE.fv(x, **param)

    @staticmethod
    def u_integral(**param):
        lmbd = param['lmbd']
        mu = param['mu']
        z = param['z']
        L = param['L']

        sqrt_mu = torch.sqrt(mu)
        EN = (lmbd / mu) * (1 - 1/torch.cosh(sqrt_mu * L / 2) * torch.cosh(0.5 * sqrt_mu * (L - 2 * z)))
        
        return EN

    @staticmethod
    def log_likelihood(x, M, **param):
        # exact data likelihood
        integral = PDE.u_integral(**param)
        u_val = PDE.fu(x, **param)
        logl = -integral * M + torch.sum(torch.log(u_val))
        return logl

    @staticmethod
    def u_exact(x, **param):
        return PDE.fu(x, **param)

    @staticmethod
    def generate_sample_thinning(N, **param):
        lmbd = param['lmbd'].item()
        mu = param['mu'].item()
        z = param['z'].item()
        L = param['L'].item()
        
        x = torch.linspace(0, L, 1000)
        u_values = PDE.fu(x, **param)
        umax = torch.max(u_values).item()

        samples_list = []

        for _ in range(N):
            n = torch.poisson(torch.tensor([umax])).int().item()

            if n == 0:
                continue

            x_rand = torch.rand(n) * L
            p = PDE.fu(x_rand, **param) / umax
            accept = torch.rand(n) < p
            samples_list.append(x_rand[accept])

        if samples_list:
            samples = torch.cat(samples_list)
        else:
            samples = torch.tensor([])

        samples = samples.view(-1, 1)
        print(f'Number of samples: {samples.shape[0]}')
        return samples


class PointProcess(BayesianProblem):
    # Poisson equation with Bayesian inference
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.input_dim = 1
        self.output_dim = 1
        self.opts=kwargs
        
        # charateristic parameters
        self.char_param = {'lmbd': 100.0, 'mu': 1.0, 'z': 1.0, 'L': 1.0}
        # setup parameters, same as Chris paper, lmbda = 500, mu = 10
        self.default_param = {'lmbd': 5.0, 'mu': 10.0, 'z': 0.5, 'L': 1.0}
        self.pde_params = ['lmbd', 'mu', 'z', 'L']
        
        # use exact solution of PDE, don't train BiLO
        self.use_exact_sol = kwargs.get('use_exact_sol', False)
        
        # use simpson's rule for computing integral
        self.use_simpson = kwargs.get('use_simpson', True)
    
        self.loss_dict['particle'] = self.nll_particle
        self.loss_dict['jump'] = self.jump_loss
        self.loss_dict['djump'] = self.djump_loss
        
        self.lambda_transform = lambda x, u, param: u**2 * x * (1 - x)

        # store jump for computing derivative of jump
        self.jump = None
        # get parameters for prior
        self.gamma_alpha = torch.tensor(kwargs['gamma'][0])
        self.gamma_beta =  torch.tensor(kwargs['gamma'][1])
        self.uniform_a =   torch.tensor(kwargs['uniform'][0])
        self.uniform_b =   torch.tensor(kwargs['uniform'][1])

        self.prior_fun = {'lmbd': self.gamma_prior, 'mu': self.soft_uniform_prior}

        self.setup_parameters(**kwargs)
        self.is_sampling = False
    
    def config_traintype(self, traintype):
        if traintype in {'bilo-simu','bilo-init','pinn-inv', 'pinn-init'}:
            self.is_sampling = False
        else:
            self.is_sampling = True

    def jump_loss(self, nn):
        # u(x, phi)
        # continuity u(z, 1) - u(z, -1) = 0
        # jump condition u_x(z, 1) - u_x(z, -1) = -lambda
        # du/dphi(z, 0) = -lambda/2
        
        x = nn.pde_params_dict['z'].view(1,1)
        # z = torch.tensor([[0.0]], requires_grad=True).to(x.device)

        u = nn(x, nn.pde_params_dict)

        # want du/dz = lambda/2
        u_z = torch.autograd.grad(u, nn.phi_a, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(u))[0]

        true_jump = - nn.params_expand['lmbd']*self.char_param['lmbd']/2
        self.jump = u_z - true_jump
        jump_loss = torch.mean((self.jump )**2) 
        return jump_loss
    
    def djump_loss(self, net):


        # djump = torch.autograd.grad(self.jump, nn., create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(self.jump))[0]
        
        grad_jump = 0.0        
        for pname in net.trainable_param:
            tmp = torch.autograd.grad(self.jump, net.params_expand[pname], grad_outputs=torch.ones_like(self.jump),
            create_graph=True, retain_graph=True,allow_unused=True)[0]
            grad_jump += torch.sum(torch.pow(tmp, 2))

        return grad_jump
    
    def residual(self, nn, x):
        
        x.requires_grad_(True)
    
        u = nn(x, nn.pde_params_dict)

        u_x = torch.autograd.grad(u, x,
            create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(u))[0]
        u_xx = torch.autograd.grad(u_x, x,
            create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(u_x))[0]
        
        res = u_xx - self.char_param['mu'] * nn.params_expand['mu'] * u
        
        return  res, u
    

    def simpson_integral(self, nn):
        # compute integral using simpson's rule with end point a and b, and exact midpoint z
        n = 100 # number of intervals
        device = nn.pde_params_dict['z'].device
        z = nn.pde_params_dict['z'].item()
        h = z / n
        x = torch.linspace(0.0, z, n+1).view(-1, 1).to(device)
        y = nn(x, nn.pde_params_dict)
        
        integral = h / 3 * (y[0] + 4 * torch.sum(y[1:-1:2]) + 2 * torch.sum(y[2:-1:2]) + y[-1])

        h = (1.0 - z) / n
        x = torch.linspace(z, 1.0, n+1).view(-1, 1).to(device)
        y = nn(x, nn.pde_params_dict)
        integral += h / 3 * (y[0] + 4 * torch.sum(y[1:-1:2]) + 2 * torch.sum(y[2:-1:2]) + y[-1])

        return integral

    def nll_particle(self, nn):
        '''compute likelihood of trajectory data'''
        xs = self.dataset['samples']
        M = self.dataset['n_snapshot']
        
        # exact integral
        if self.use_simpson:
            integral = self.simpson_integral(nn)
        else:
            # use exact integral
            integral = PDE.u_integral(lmbd = nn.pde_params_dict['lmbd']*self.char_param['lmbd'], mu = nn.pde_params_dict['mu']*self.char_param['mu'], z = nn.pde_params_dict['z'], L = nn.pde_params_dict['L'])
        
        # squeze integral
        integral = torch.squeeze(integral)
        
        u = nn(xs, nn.pde_params_dict)
        ll = - M * integral + torch.sum(torch.log(u))
        nll = -ll
        return nll

    def uniform_prior(self, x):
        a = self.uniform_a
        b = self.uniform_b
        p = 1.0/(b-a)
        
        P = torch.where(( x >= a) & (x <= b), p , torch.tensor(0.0))
        logP = torch.log(P)
        return logP
    
    def soft_uniform_prior(self, x):
        penalty_scale = 1e3
        a = self.uniform_a.to(x.device)
        b = self.uniform_b.to(x.device)

        inside_mask = (x >= a) & (x <= b)
        outside_upper = x > b
        outside_lower = x < a
        
        # Probability inside
        p_inside = 1.0 / (b - a)
        
        # Distance-based penalties outside
        dist_outside = torch.relu(x - b) + torch.relu(a - x)
        p_outside = torch.exp(-penalty_scale * dist_outside)
        # Combine
        p = torch.where(inside_mask, p_inside, p_outside)

        epsilon = 1e-12
        return torch.log(p + epsilon)

    def gamma_prior(self, x):
        # use definition from wiki, with alpha = shape, beta = rate
        # mean is alpha/beta, variance is alpha/beta^2
        alpha = self.gamma_alpha 
        beta = self.gamma_beta

        # gamma_alpha = torch.exp(torch.lgamma(alpha))
        # P = torch.exp(-beta * x) * beta**alpha / gamma_alpha * x**(alpha - 1)
        # logP = torch.log(P)

        log_gamma_alpha = torch.lgamma(alpha)  # Log of Gamma(alpha)
        logP = -beta * x + alpha * torch.log(beta) - log_gamma_alpha + (alpha - 1) * torch.log(x)
        
        return logP

    def nll_prior_pde(self, nn):
        '''P(\Theta), prior of PDE parameter'''
        # # uniform distribution in [a, b]
        # a = 0
        # b = self.gt_param['mu'] * 2
        # mu = nn.pde_params_dict['mu']
        # P = torch.where(( mu >= a) & (mu <= b), torch.tensor(1.0/(b-a)), torch.tensor(0.0))
        # logP = torch.log(P)
        
        logP = 0.0
        for pname in nn.trainable_param:
            logP += self.prior_fun[pname](nn.pde_params_dict[pname] * self.char_param[pname])

        nll = -logP.squeeze()
        return nll


    
    def u_exact(self, x, param:dict):
        return PDE.u_exact(x, lmbd = param['lmbd'] * self.char_param['lmbd'], mu = param['mu'] * self.char_param['mu'], z = param['z'], L = param['L'])

    def print_info(self):
        # print info of pde
        # print all parameters
        pass
    
    def setup_network(self, **kwargs):
        '''setup network, get network structure if restore'''
        # then update by init_param if provided
        kwargs['input_dim'] = self.input_dim + 1 # extra dimension for phi(x)
        kwargs['output_dim'] = self.output_dim


        if self.use_exact_sol:
            net = ExactSolution(**kwargs,
                                lambda_transform = self.lambda_transform,
                                all_params_dict = self.init_param,
                                pde_params=self.pde_params,
                                trainable_param = self.opts['trainable_param'],
                                char_param = self.char_param)
            net.param_net = []
        else:
            net = CuspNet(**kwargs,
                            lambda_transform = self.lambda_transform,
                            all_params_dict = self.init_param,
                            pde_params=self.pde_params,
                            trainable_param = self.opts['trainable_param'],
                            char_param = self.char_param)
        return net



    def setup_dataset(self, dsopt, noise_opts=None, device='cuda'):
        # create dataset from pde using datset option and noise option
        self.dataset = MatDataset()

        n_snapshot = dsopt['n_snapshot']

        # Collocation points for residual loss
        self.dataset['X_res_train'] = torch.linspace(0, 1, dsopt['N_res_train']).view(-1, 1)
        self.dataset['X_res_test'] =  torch.linspace(0, 1, dsopt['N_res_test']).view(-1, 1)
 
        # remove x close to z for training, already in the cusp condition
        # tol = 0.01
        # mask_away = torch.abs(self.dataset['X_res_train'] - self.gt_param['z']) > tol
        # self.dataset['X_res_train'] = self.dataset['X_res_train'][mask_away].view(-1, 1)
        # self.dataset['N_res_train'] = self.dataset['X_res_train'].shape[0]
        
        # data for trainig
        self.dataset['X_dat_train'] = torch.linspace(0, 1, dsopt['N_dat_train']).view(-1, 1)
        self.dataset['X_dat_test'] =  torch.linspace(0, 1, dsopt['N_dat_test']).view(-1, 1)

        self.dataset['u_dat_test'] = self.u_exact(self.dataset['X_dat_test'], self.gt_param)
        self.dataset['u_dat_train'] = self.u_exact(self.dataset['X_dat_train'], self.gt_param)

        
        self.dataset['samples'] = PDE.generate_sample_thinning(n_snapshot, lmbd = self.gt_param['lmbd']*self.char_param['lmbd'],
         mu = self.gt_param['mu']*self.char_param['mu'], z = self.gt_param['z'], L = self.gt_param['L'])
        self.dataset['n_snapshot'] = n_snapshot
    
        self.dataset.to_device(device)

    @torch.no_grad()
    def validate(self, nn):
        '''compute err '''
        # compare prediction with exact solution
        d = {}
        
        # linf_err between NN and exact solution 
        prediction = nn(self.dataset['X_dat_test'], nn.pde_params_dict)
        u_exact = self.u_exact(self.dataset['X_dat_test'], nn.pde_params_dict)
        linf_err = torch.max(torch.abs(prediction - u_exact)).item()
        d['linf_err'] = linf_err

        for pname in nn.trainable_param:
            d[pname] = nn.pde_params_dict[pname].item()

        # exact likelihood
        exact_nll = -PDE.log_likelihood(self.dataset['samples'], self.dataset['n_snapshot'], \
            lmbd = nn.pde_params_dict['lmbd']*self.char_param['lmbd'],\
            mu = nn.pde_params_dict['mu']*self.char_param['mu'],\
            z = nn.pde_params_dict['z'],\
            L = nn.pde_params_dict['L'])
        
        approx_nll = self.nll_particle(nn)

        nll_error = approx_nll - exact_nll
        d['nll_error'] = nll_error.item()

        # for sampling, compute mean and std of parameters
        if self.is_sampling:
            for pname in nn.trainable_param:
                mean = self.estimator.get_mean(pname)
                std = torch.sqrt(self.estimator.get_population_variance(pname))
                d[pname + '_mean'] = mean.item()
                d[pname + '_std'] = std.item()
        
        return d
    
    @error_logging_decorator
    def visualize_upred_histogram(self, savedir=None):
        # plot predicted u and the histogram of samples

        # check if sasmples are generated
        if 'samples' not in self.dataset:
            return 
        

        # scatter plot of training data, might be noisy
        x_dat_train = self.dataset['X_dat_train']
        u_dat_train = self.dataset['u_dat_train']

        # line plot gt solution and prediction
        u_test = self.dataset['u_dat_test']
        upred = self.dataset['upred_dat_test']
        x_dat_test = self.dataset['X_dat_test']

        # visualize the results
        fig, ax = plt.subplots()

        # plot histogram of samples as probability density
        samples = self.dataset['samples']
        ax.hist(samples, bins=20, density=True, alpha=0.6, color='g', label='samples')

        # plot prediction
        expectation = PDE.u_integral(lmbd=self.gt_param['lmbd']*self.char_param['lmbd'], mu=self.gt_param['mu']*self.char_param['mu'], z=self.gt_param['z'], L=self.gt_param['L'])
        ax.plot(x_dat_test, upred/expectation, label='NN prediction')

        if savedir is not None:
            fpath = os.path.join(savedir, 'fig_pred_histo.png')
            fig.savefig(fpath, dpi=300, bbox_inches='tight')
            print(f'fig saved to {fpath}')

        return fig, ax

    def visualize(self, savedir=None):
        # visualize the results
        self.dataset.to_np()

        self.plot_prediction(savedir)
        self.plot_variation(savedir)

        if self.is_sampling:
            self.plot_mean_std(savedir)
            self.visualize_distribution(savedir)
            self.visualize_upred_histogram(savedir)

        
if __name__ == "__main__":
    import sys
    from Options import *
    from DenseNet import *
    # from Problems import 


    optobj = Options()
    optobj.opts['pde_opts']['problem'] = 'poisson'

    optobj.parse_args(*sys.argv[1:])
    
    device = set_device('cuda')
    set_seed(0)
    
    optobj.print()

    prob = PointProcess(**optobj.opts['pde_opts'])
    pdenet = prob.setup_network(**optobj.opts['nn_opts'])
    pdenet.to(device)
    prob.setup_dataset(optobj.opts['dataset_opts'], optobj.opts['noise_opts'], device=device)

    prob.make_prediction(pdenet)
    prob.visualize(savedir='tmp')

    # save dataset
    fpath = os.path.join('tmp', 'dataset.mat')
    prob.dataset.save(fpath)
