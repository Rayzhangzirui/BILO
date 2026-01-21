#!/usr/bin/env python

from typing import Iterable
from abc import abstractmethod
import time

import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from loralib import mark_only_lora_as_trainable
from utillora import reset_lora_weights, merge_lora_weights
from Trainer import *

from torch.nn.utils.convert_parameters import parameters_to_vector, vector_to_parameters

from lossCollection import lossCollection, EarlyStopping
from BayesianProblem import WelfordEstimator
from Logger import Logger

import logging
import sys

# Create a logger
level = logging.INFO
log = logging.getLogger('sampler')
log.setLevel(level)
# Create handlers
console_handler = logging.StreamHandler(sys.stdout)
# Set logging level for handlers
console_handler.setLevel(level)
# Create a formatter and set it for both handlers
formatter = logging.Formatter('%(message)s')
console_handler.setFormatter(formatter)
# Add handlers to the logger
log.addHandler(console_handler)



# generator for momentum variable
SAMPLE_GENERATOR = None

# temporary function to log memory usage
def log_memory(prefix=""):
    # torch.cuda.synchronize()
    # allocated = torch.cuda.memory_allocated()
    # reserved = torch.cuda.memory_reserved()
    # peak = torch.cuda.max_memory_allocated()
    # print(f"{prefix}Allocated: {allocated/1e6:.2f} MB, Reserved: {reserved/1e6:.2f} MB, Peak: {peak/1e6:.2f} MB")
    # # Reset peak stats if needed
    # torch.cuda.reset_peak_memory_stats()
    pass

def log_dict(indent, **kwargs):
    processed_items = []
    
    for k, v in kwargs.items():
        if isinstance(v, torch.Tensor):
            # Convert PyTorch tensors to native Python values
            v = v.detach().cpu().numpy() if v.requires_grad else v.cpu().numpy()
            if v.size == 1:  # If tensor is scalar-like
                v = v.item()
        elif isinstance(v, float):
            # Format floats in scientific notation
            v = f'{v:.3e}'
        elif isinstance(v, nn.Module):
            continue
        
        processed_items.append(f'{k}: {v}')
    
    combine_str = ', '.join(processed_items)
    log.debug('\t' * indent + combine_str)    

class Sampler(Trainer):
    # same constructor as Trainer
    def __init__(self, *args):
        super().__init__(*args)

        self.skip_lower = self.pde.use_exact_sol
        self.debug = False

        if not self.skip_lower:
            # optimizer and scheduler for lower level
            optimizer_net = optimizer_dictionary[self.opts['optim_net']]
            optim_net_opts = self.opts['opts_net']
            
            if self.net.rank == 0 or self.opts['loraplus_lr_ratio'] == 1:
                self.optimizer['param_net'] = optimizer_net(self.net.param_net, lr=self.opts['lr_net'], **optim_net_opts)
            
            else:
                # try lora+
                # collect groupA and groupB
                groupA_params = []
                groupB_params = []
                for name, param in self.net.named_parameters():
                    if not param.requires_grad:
                        continue
                    # Put parameters in groupB if the name indicates it's a LoRA parameter or if it is 1-dimensional.
                    if "lora_B" in name or param.ndim == 1:
                        groupB_params.append(param)
                    else:
                        groupA_params.append(param)
                
                base_lr = self.opts['lr_net']
                loraplus_lr_ratio = self.opts['loraplus_lr_ratio']
                optimizer_grouped_parameters = [
                    {"params": groupA_params, "lr": base_lr},
                    {"params": groupB_params, "lr": base_lr * loraplus_lr_ratio},
                ]

                self.optimizer['param_net'] = optimizer_net(optimizer_grouped_parameters, **optim_net_opts)

            if self.opts['sch_warmup'] == True:

                scheduler_net = getattr(optim.lr_scheduler, self.opts['sch_net'])
                main_scheduler = scheduler_net(self.optimizer['param_net'], **self.opts['schopt_net'])
            
                warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                    self.optimizer['param_net'],
                    start_factor=0.1,      # lr starts at 0.1 * base_lr
                    total_iters=100        # for the first 100 calls to step()
                )

                self.scheduler['param_net'] = torch.optim.lr_scheduler.SequentialLR(
                    self.optimizer['param_net'],
                    schedulers=[warmup_scheduler, main_scheduler],
                    milestones=[100]
                )
            else:
                scheduler_net = getattr(optim.lr_scheduler, self.opts['sch_net'])
                self.scheduler['param_net'] = scheduler_net(self.optimizer['param_net'], **self.opts['schopt_net'])


        

        # Setup parameters for sampling
        if self.net.with_param is True:
            # bilo network
            self.sample_param = self.net.param_pde_trainable
        else:
            # PINN
            self.sample_param = self.net.param_all
            self.skip_lower = True
        
        self.W_current = None
        self.w_proposal = None

        self.x_current = None
        self.x_proposal = None

        self.grad_current = None
        self.grad_proposal = None

        self.w_upper_current = None
        self.w_upper_proposal = None

        # if lower_bound and upper_bound are specified in pde, use them
        if hasattr(self.pde, 'lower_bound'):
            self.lower_bound = torch.tensor([self.pde.lower_bound.get(k, -1e6) for k in self.net.trainable_param], device=self.device)
            self.upper_bound = torch.tensor([self.pde.upper_bound.get(k, 1e6) for k in self.net.trainable_param], device=self.device)
        else:
            self.lower_bound = torch.tensor([-1e6]*len(self.net.trainable_param), device=self.device)
            self.upper_bound = torch.tensor([1e6]*len(self.net.trainable_param), device=self.device)

        self.total_step = 0
        # total step increses whenever W or \theta changes
        self.pde.estimator.burnin = self.opts['burnin']

        # array of standard deviation
        # check size of std_mx and number of parameters, padd with 1.0 if not enough
        if len(self.opts['std_mx']) < len(self.net.trainable_param):
            self.opts['std_mx'] = self.opts['std_mx'] + [1.0] * (len(self.net.trainable_param) - len(self.opts['std_mx']))
        elif len(self.opts['std_mx']) > len(self.net.trainable_param):
            self.opts['std_mx'] = self.opts['std_mx'][:len(self.net.trainable_param)]
        logging.info(f"Using std_mx: {self.opts['std_mx']} for sampling.")
        self.std_matrix = torch.tensor(self.opts['std_mx'], device=self.device) # initial std for each parameter
        self.variance_estimator = WelfordEstimator(burnin=0)

        SAMPLE_GENERATOR = torch.Generator(self.device)
        SAMPLE_GENERATOR.manual_seed(self.opts['sample_seed'])
        

    @abstractmethod
    def get_proposal(self):
        ''' get proposal for the next step '''
        pass

    @abstractmethod
    def get_prob_dict(self):
        ''' get acceptance probability '''
        pass

    # For bilevel
    def get_lower_loss(self, yes_grad = True):
        # self.optimizer['param_net'].zero_grad()
        weighted_sum, weighted_loss_comp, unweighted_loss_comp = self.lossCollection.get_wloss_sum_comp(self.loss_net, yes_grad)
        return weighted_sum, weighted_loss_comp, unweighted_loss_comp
        
    def step_lower(self, loss_lower):
        self.set_grad(self.net.param_net, loss_lower)
        self.optimizer['param_net'].step()
        self.scheduler['param_net'].step()
    
    def get_upper_loss(self, yes_grad = True):
        weighted_sum, weighted_loss_comp, unweighted_loss_comp = self.lossCollection.get_wloss_sum_comp(self.loss_pde, yes_grad)
        return weighted_sum, weighted_loss_comp, unweighted_loss_comp

    def lower_level_loop(self):
        # do lower level loop until convergence
        
        at_lower = 1
        stophere = False

        lower_step_counter = 0 # local counter for lower level
        log.debug('\n')

        if 'total_lower_time' not in self.info:
            self.info['total_lower_time'] = 0
        start = time.time()
        
        while True:

            w_lower, w_lower_comp, uw_lower_comp = self.get_lower_loss(True)
            
            # if self.opts['skip_lower_log'] == False:
            #     self.log_stat(self.total_step, stophere, **uw_lower_comp, **uw_upper_comp, lower=at_lower, lowertot=w_lower, uppertot=w_upper)
            #     self.validate(self.total_step, stophere)

            log_dict(2,lowerstep=lower_step_counter, w_lower=w_lower, **w_lower_comp, **self.net.all_params_dict)
            if w_lower < self.opts['tol_lower']:
                # log_dict(2,lowerstep=lower_step_counter, w_lower=w_lower, **w_lower_comp, **self.net.all_params_dict)
                break
            
            self.set_grad(self.net.param_net, w_lower)
            self.optimizer['param_net'].step()
            self.scheduler['param_net'].step()
            self.optimizer['param_net'].zero_grad()

            lower_step_counter += 1

            # if self.opts['skip_lower_log'] == False:
            #     self.total_step += 1

            if lower_step_counter == self.opts['max_iter_lower']:
                break
        
        end = time.time()
        self.info['total_lower_time'] += end - start

        
        # state include lower level component, lower = total lower level loss, and the number of steps
        state = {}
        state.update(uw_lower_comp)
        # detach all tensors
        state = {k: v.detach() for k, v in state.items()}
        # collect lower level info
        state['lower_step'] = lower_step_counter
        state['lower'] = w_lower
        return state


    def get_energy_grad(self,x):
        vector_to_parameters(x, self.sample_param)
        # w_upper = - log p(theta|D), which is the potential energy
        w_upper, w_upper_comp, uw_upper_comp = self.get_upper_loss(True)
        # grad_current = - grad log p
        grad = parameters_to_vector(torch.autograd.grad(w_upper, self.sample_param, retain_graph=False))

        # detach all tensors
        w_upper = w_upper.detach()
        w_upper_comp = {k: v.detach() for k, v in w_upper_comp.items()}
        uw_upper_comp = {k: v.detach() for k, v in uw_upper_comp.items()}
        grad = grad.detach()
        return w_upper, w_upper_comp, uw_upper_comp, grad
    


    def train_loop(self):
        # Metropolis-Hasting
        self.x_current = parameters_to_vector(self.sample_param)

        num_accept = 0
        upper_step = 0
        # zero grad for all parameters
        if not self.skip_lower:
            self.W_current = parameters_to_vector(self.net.param_net)
            self.optimizer['param_net'].zero_grad()
            lower_state = self.lower_level_loop()
        
        # initial state, 
        # self.w_upper_current, w_upper_comp_current, uw_upper_comp_current, self.grad_current = self.get_energy_grad(self.x_current)
        # current_param = {k:v.clone().detach() for k,v in self.net.all_params_dict.items()}

        min_upper = 1e9
        
        if self.opts['backtrack']:
            saved_opt_state = copy.deepcopy(self.optimizer['param_net'].state_dict())
        
        while True:
            log_memory(f"Iteration {self.total_step} start: ")
            # generate proposal

            # current parameter
            cr_param = {k: self.net.all_params_dict[k].clone().detach() for k in self.net.trainable_param if isinstance(self.net.all_params_dict[k], nn.Parameter)}
            
            # get proposal
            lower_state = self.get_proposal()
            
            # proposed parameter
            pp_param = {k: self.net.all_params_dict[k].clone().detach() for k in self.net.trainable_param if isinstance(self.net.all_params_dict[k], nn.Parameter)}
            
            # get proposal loss
            self.w_upper_proposal, self.w_upper_comp_proposal, self.uw_upper_comp_proposal, self.grad_proposal = self.get_energy_grad(self.x_proposal)
            
            # if proposal is nan, reject
            accept =  ~ torch.isnan(self.w_upper_proposal)
                
            # compute acceptance probability
            prob_dict = self.get_prob_dict()
            
            rand = torch.rand(1, device=self.device, generator=SAMPLE_GENERATOR)
            accept = accept & ( rand < prob_dict['prob'])
            
            ### accept or reject
            if accept:
                # accept
                accept = 1
                
                self.grad_current = self.grad_proposal
                self.w_upper_current = self.w_upper_proposal
                self.x_current = self.x_proposal


                # merge and reset seems to slow down the training
                if self.opts['merge']:
                    merge_lora_weights(self.net)
                    reset_lora_weights(self.net)
                    # reset optimizer
                    self.optimizer['param_net'] = optim.Adam(self.net.param_net, lr=self.opts['lr_net'])

                num_accept += 1
                if self.opts['backtrack']:
                    saved_opt_state = copy.deepcopy(self.optimizer['param_net'].state_dict())
            else:
                accept = 0
                
                vector_to_parameters(self.x_current, self.sample_param)

                # reset network weight
                if not self.skip_lower:
                    vector_to_parameters(self.W_current, self.net.param_net)
                
                if self.opts['backtrack']:
                    self.optimizer['param_net'].load_state_dict(saved_opt_state)
            ### End of accept or reject
            
            # this remove the internal state 
            if self.opts['refresh']:
                self.optimizer['param_net'] = optim.Adam(self.net.param_net, lr=self.opts['lr_net'])

            log_memory(f"Iteration {self.total_step} end: ")

            self.total_step += 1
            stophere = self.estop(self.total_step, None)
            if stophere:
                break
            
            upper_step += 1
            # log acceptance rate
            accept_rate = num_accept / upper_step
            self.pde.update_estimator(self.net)

            # streaming estimate of sample variance
            if self.opts['adapt_M']:
                if self.total_step < self.opts['burnin']:
                    self.variance_estimator.update(self.x_current,'x')
                # set sample variance
                if self.total_step == self.opts['burnin']:
                    v = torch.sqrt(self.variance_estimator.get_variance('x')+1e-6)
                    # rescale v to be order 1.
                    self.std_matrix = v/torch.mean(v)
                    
            

            # print MH stats: energy and acceptance rate
            if self.total_step % self.opts['print_every'] == 0 or stophere:
                # print current and proposal
                self.logger.log_metrics({'upper':self.w_upper_current}, step=self.total_step, prefix='cr_')
                self.logger.log_metrics({'upper':self.w_upper_proposal}, step=self.total_step, prefix='pp_')
                self.logger.log_metrics(self.uw_upper_comp_current, step=self.total_step, prefix='cr_')
                self.logger.log_metrics(self.uw_upper_comp_proposal, step=self.total_step, prefix='pp_')
                self.logger.log_metrics(cr_param, step=self.total_step, prefix='cr_')
                self.logger.log_metrics(pp_param, step=self.total_step, prefix='pp_')
                
                # print proposed parameters
                # self.logger.log_metrics(proposed_param, step=self.total_step, prefix='pp')

                # print probablity
                self.logger.log_metrics(prob_dict, step=self.total_step)

                if not self.skip_lower:
                    self.logger.log_metrics(lower_state, step=self.total_step)

                # print MH decision
                self.logger.log_metrics({'acc':accept, 'acrate':accept_rate}, step=self.total_step)


            # Collect solution and MAP
            # The solution is correct only for newly accepted step. 
            # For rejected step, need to run lower level loop, currently handled inside get_proposal
            if accept:
                if self.opts['example_every']>=1 and (num_accept % self.opts['example_every'] == 0):
                    self.pde.collect_solution(self.net)
                
                # if min_upper decrease, update and collect MAP
                if self.w_upper_current < min_upper:
                    min_upper = self.w_upper_current
                    self.pde.collect_solution_MAP(self.net)

            # print estiamted mean and variance of parameter
            self.validate(self.total_step, stophere)

            # if accept_rate too low after 100 steps, break
            if upper_step > 100 and accept_rate < self.opts['acc_threshold']:
                break
            # next cycle

        
        ### end of metropolis-hasting

class MetropolisAdjustedLangevinDynamics(Sampler):
    def __init__(self, *args):
        super().__init__(*args)

    
    def get_proposal(self):
        # generate proposal
        # this part is obsolete, throw exception
        # need to run lower level for x_current
        raise NotImplementedError("MALA Need update.")
        size = self.x_current.size()
        z = torch.randn(size, device=self.device, generator=SAMPLE_GENERATOR)
        self.x_proposal = self.x_current - self.grad_current * self.opts['lr_pde'] + z * torch.sqrt(2 * torch.tensor(self.opts['lr_pde']))

        # map back to parameters
        vector_to_parameters(self.x_proposal, self.sample_param)
        if not self.skip_lower:
            lower_state = self.lower_level_loop()
    
    def get_prob_dict(self):
        prob_current_to_proposal = (-1/(4*self.opts['lr_pde'])) * torch.norm(self.x_proposal - self.x_current + self.opts['lr_pde'] * self.grad_current) ** 2
        prob_proposal_to_current = (-1/(4*self.opts['lr_pde'])) * torch.norm(self.x_current - self.x_proposal + self.opts['lr_pde'] * self.grad_proposal) ** 2
        
        H_current = -self.w_upper_current + prob_current_to_proposal
        H_proposal = -self.w_upper_proposal + prob_proposal_to_current
        
        # delta_H = H_proposal - H_current
        # prob = torch.exp(-delta_H)

        prob = torch.exp(- self.w_upper_proposal + self.w_upper_current + (prob_proposal_to_current - prob_current_to_proposal))

        log_dict(0, prob=prob.item(), wc=self.w_upper_current.item(), wp=self.w_upper_proposal.item(), prob_c2p=prob_current_to_proposal.item(), prob_p2c=prob_proposal_to_current.item())
        # return {'prob':prob, 'wc':self.w_upper_current, 'wp':self.w_upper_proposal, 'prob_c2p':prob_current_to_proposal, 'prob_p2c':prob_proposal_to_current}
        return {'prob':prob}


class HamiltonianMonteCarlo(Sampler):
    def __init__(self, *args):
        super().__init__(*args)
        
        self.lf_step_size = torch.tensor(self.opts['lr_pde'], device=self.device)
        self.lf_steps = self.opts['lf_steps']

        # Will store current momentum / proposal momentum
        self.p_init = None
        self.p_proposal = None

        self.info['total_lower_steps'] = 0
    
    def get_proposal(self):
        # 1) Sample random momentum
        size = self.x_current.size()
        self.p_init = self.std_matrix * torch.randn(size, device=self.device, generator=SAMPLE_GENERATOR)

        # for debug, all 0
        # self.p_init = torch.zeros_like(self.x_current, device=self.device)
        
        x = self.x_current
        p = self.p_init
    
        lower_state = {}
        log.debug('\n')
        log_dict(1, lf_step=0, x=x.detach().cpu().numpy(), p=p.detach().cpu().numpy())
        
        prop_lower_steps = 0 # total lower steps for this proposal
            
        step_size = self.lf_step_size
        
        # draw ranodm L from [1, 2*lf_steps]
        if self.opts['random_L']:
            L = torch.randint(1, 2*self.lf_steps, (1,), generator=SAMPLE_GENERATOR).item()
        else:
            L = self.lf_steps

        if not self.skip_lower:
            lower_state = self.lower_level_loop()
            prop_lower_steps += lower_state['lower_step']
        
        self.w_upper_current, self.w_upper_comp_current, self.uw_upper_comp_current, self.grad_current = self.get_energy_grad(self.x_current)

        for i in range(1, L+1):
            # recompute grad at new x

            w_temp,_ , _, grad_x = self.get_energy_grad(x)
            # half step for p
            if i == 1:
                p = p - 0.5 * step_size * grad_x
            else:
                p = p - step_size * grad_x

            # full step for x
            x = x + step_size * p/(self.std_matrix**2)
            
            # for debuging 
            # w_lower, w_lower_comp, uw_lower_comp = self.get_lower_loss(True)
            log_dict(1,lf_step=i, x=x.detach().cpu().numpy(), p=p.detach().cpu().numpy(), grad_x=grad_x.detach().cpu().numpy())

            # implement the billard scheme for boundary conditions
            # if x is out of bounds, reflect the momentum
            x = torch.where(x < self.lower_bound, 2 * self.lower_bound - x, x)
            p = torch.where(x < self.lower_bound, -p, p)

            x = torch.where(x > self.upper_bound, 2 * self.upper_bound - x, x)
            p = torch.where(x > self.upper_bound, -p, p)
            
            # Update x
            # Here the Hamiltonian is exact and depends on x.
            # Might need to run lower level here. 
            vector_to_parameters(x, self.sample_param)
            if not self.skip_lower:
                lower_state = self.lower_level_loop()
                prop_lower_steps += lower_state['lower_step']

            # last half step for p if we are on the last iteration
            if i == self.lf_steps:
                _, _, _, grad_x = self.get_energy_grad(x)
                p = p - 0.5 * step_size * grad_x
            
        
        # store final x, p in the proposal
        self.x_proposal = x
        self.p_proposal = p
        # store total lower steps
        if not self.skip_lower:
            lower_state['lower_step'] = prop_lower_steps
            self.info['total_lower_steps'] += prop_lower_steps
        return lower_state

    def get_prob_dict(self):
        """
        Compute acceptance probability = exp(-[H_proposal - H_current]).
        """
        # Current Hamiltonian
        kenetic_current = 0.5 * torch.dot(self.p_init, self.p_init/(self.std_matrix**2))
        H_current = self.w_upper_current + kenetic_current

        # Proposed Hamiltonian
        kenetic_proposal = 0.5 * torch.dot(self.p_proposal, self.p_proposal/(self.std_matrix**2))
        H_proposal = self.w_upper_proposal + kenetic_proposal

        # ΔH
        delta_H = H_proposal - H_current

        # acceptance prob
        prob = torch.min(torch.exp(-delta_H), torch.tensor(1.0, device=self.device))
        # print(f"prob: {prob.item()}, w_current: {self.w_upper_current.item()}, k_current: {kenetic_current.item()}, w_proposal: {self.w_upper_proposal.item()}, k_proposal: {kenetic_proposal.item()}, delta_H: {delta_H.item()}")
        # return {'prob':prob, 'wc':self.w_upper_current, 'wp':self.w_upper_proposal, 'kc':kenetic_current, 'kp':kenetic_proposal}
        return {'prob':prob, 'Hc':H_current, 'Hp':H_proposal, 'dH':delta_H}



if __name__ == "__main__":
    # test sampling using simple 2d distributions
    
    from Options import Options
    from BayesianProblem import BayesianProblem
    from util import set_device, set_seed, print_dict
    import sys
    import matplotlib.pyplot as plt

    class ToyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.param_net = []  # empty in this toy example
            # Start near (0.5, -0.5)
            self.all_params_dict = nn.ParameterDict({
                'x':torch.nn.Parameter(torch.tensor(0.5)),
                'y':torch.nn.Parameter(torch.tensor(-0.5))
            })
            self.param_pde_trainable = list(self.all_params_dict.values())
            self.pde_params_dict = self.all_params_dict
            self.trainable_param = ['x', 'y']
    
    class ToyDist(BayesianProblem):
        def __init__(self):
            super(ToyDist, self).__init__()
            self.loss_dict = {}
            self.loss_dict['energy'] = self.get_energy
            self.use_exact_sol = True
            self.hist = {'x':[], 'y':[]}
        
        def energy(self, x, y):
            pass

        def get_energy(self, net):
            x = net.all_params_dict['x']
            y = net.all_params_dict['y']
            return self.energy(x, y)
        
        def residual(self, net, x):
            return 0.0
        
        def setup_dataset(self, nn, x):
            return 0.0
        
        def plot_samples(self, savedir = None):
            x = self.hist['x']
            y = self.hist['y']

            # contour plot of the energy
            xx, yy = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
            # convert to torch tensor
            xx = torch.tensor(xx, dtype=torch.float32)
            yy = torch.tensor(yy, dtype=torch.float32)
            zz = self.energy(xx, yy)

            fig, ax = plt.subplots()

            ax.contour(xx, yy, zz, 20)
            ax.scatter(x, y, color='red', s=1)

            if savedir is not None:
                fpath = os.path.join(savedir, f'fig_scatter.png')
                fig.savefig(fpath, dpi=300, bbox_inches='tight')
                print(f'fig saved to {fpath}')
    
    class Gaussian(ToyDist):
        def energy(self, x, y):
            return 0.5 * (x ** 2 + y ** 2)
    
    class Banana(ToyDist):
        def energy(self, x, y):
            # rosenbrock function
            # Eq (2) in "An n-dimensional Rosenbrock Distribution for MCMC Testing"
            a = 1
            b = 100
            y = (a - x) ** 2 + b * (y - x ** 2) ** 2
            y = y/20
            # return negative log likelihood
            return y
            

    optobj = Options()
    # change default for testing
    optobj.opts['flags'] = 'local'
    optobj.opts['train_opts']['loss_pde']= 'energy'
    optobj.parse_args(*sys.argv[1:])
     
    device = set_device('cuda')
    set_seed(0)

    # setup pde/dist
    dist_dict = {'gaussian': Gaussian, 'banana': Banana}
    dist_name = optobj.opts['pde_opts']['problem']
    pde = dist_dict[dist_name]()

    # setup logger
    logger = Logger(optobj.opts['logger_opts']) 

    # setup network
    net = ToyNet()
    net.to(device)

    # set up los
    lc = lossCollection(net, pde, {'energy': 1.0})

    print_dict(optobj.opts)
    ### basic
    
    trainer_dictionary = {
    'mala':MetropolisAdjustedLangevinDynamics,
    'hmc': HamiltonianMonteCarlo,
    }
    

    sampler_name = optobj.opts['traintype']
    sampler = trainer_dictionary[sampler_name]

    trainer = sampler(optobj.opts['train_opts'], net, pde, device, lc, logger)
    trainer.train()

    pde.visualize_distribution(savedir = 'runs/tmp/')
    
    # scatter plot of samples
    pde.plot_samples(savedir = 'runs/tmp/')






    