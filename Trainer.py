#!/usr/bin/env python

# for training the network
# need: options, network, pde, dataset, lossCollection
import os
import abc
import time
import torch
import torch.optim as optim
import abc
from lossCollection import lossCollection, EarlyStopping
from Logger import Logger
from util import get_mem_stats, set_device, set_seed, print_dict, flatten

from SGLD import SGLD
from precondSGLD import pSGLD
from PyTorch_LBFGS.functions.LBFGS import FullBatchLBFGS

from loralib import mark_only_lora_as_trainable
from utillora import reset_lora_weights, merge_lora_weights

import torch.profiler

import copy

optimizer_dictionary = {
    'SGD': optim.SGD,
    'Adam': optim.Adam,
    'AdamW': optim.AdamW,
    'lbfgs': optim.LBFGS,
    'SGLD': SGLD,
    'pSGLD': pSGLD
}



class Trainer(abc.ABC):
    def __init__(self, opts, net, pde, device, lossCollection, logger:Logger):
        self.opts = opts
        self.logger = logger
        self.net = net
        self.pde = pde
        self.device = device
        self.info = {}

        
        self.lossCollection = lossCollection
        
        self.optimizer = {}
        self.scheduler = {}

        # early stopping
        self.estop = EarlyStopping(**self.opts)

        self.loss_net = opts['loss_net']
        self.loss_pde = opts['loss_pde']
        self.loss_test = opts['loss_test']
        self.loss_monitor = opts['loss_monitor']

    
    def log_trainable_params(self, step):
        '''
        log trainable parameters
        '''
        for key in self.net.trainable_param:
            self.logger.log_metrics({key:self.net.all_params_dict[key].item()}, step=step)
        
    def log_stat(self, step, stophere, **kwargs):
        # kwargs is key value pair
        if step % self.opts['print_every'] == 0 or stophere:
            for key, val in kwargs.items():
                if val is not None:
                    self.logger.log_metrics({key:val}, step=step)
        
            # if not self.net.with_func:
            #     # log network parameters if param not function
            #     for key in self.net.trainable_param:
            #         self.logger.log_metrics({key:self.net.all_params_dict[key].item()}, step=step)
    
    @torch.no_grad
    def validate(self, epoch, stophere):
        if epoch % self.opts['print_every'] == 0 or stophere:
            val = self.pde.validate(self.net)
            self.logger.log_metrics(val, step=epoch)
            
            # log testing loss
            wtotal, wloss_comp, uwloss_comp = self.lossCollection.get_wloss_sum_comp(self.loss_test, False)
            self.logger.log_metrics(uwloss_comp, step=epoch)
    
    def set_grad(self, params, loss):
        '''
        set gradient of loss w.r.t params
        '''
        grads = torch.autograd.grad(loss, params, retain_graph=False)
        for param, grad in zip(params, grads):
            param.grad = grad
    
    def acc_grad(self, params, loss):
        '''
        accumulate gradient of loss w.r.t params
        '''
        # skip if loss is None, this can happen if upper loss is not specified, e.g. during pre-training for scalar problem
        if loss is None:
            return
        
        grads = torch.autograd.grad(loss, params, retain_graph=False)
        for param, grad in zip(params, grads):
            # skip if grad is None
            if grad is None:
                continue
            # if param.grad is None, then initial grad, otherwise accumulate
            if param.grad is None:
                param.grad = grad
            else:
                param.grad += grad

    def train(self):
        # move to device

        self.info['num_params'] = sum(p.numel() for p in self.net.parameters())
        self.info['num_train_params'] = sum(p.numel() for p in self.net.parameters() if p.requires_grad)

        self.net.to(self.device)
        # self.pde.dataset.device = self.device
        # self.pde.dataset.to_device(self.device)

        start = time.time()

        # reset memeory usage
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_max_memory_allocated()
        
        # for running
        try:    
            self.train_loop()
        except KeyboardInterrupt:
            print('Interrupted by user')
            self.info.update({'error':'interrupted by user'})
            self.logger.set_tags('status', 'FAILED')
        except Exception as e:
            self.info.update({'error':str(e)})
            # mlflow set state to failed
            print(f'Error: {str(e)}')
            self.logger.set_tags('status', 'FAILED')
            raise e
            

        # for profiling
        # writer = torch.utils.tensorboard.SummaryWriter(log_dir='./log/tmp')

        # with torch.profiler.profile(
        #         activities=[
        #             torch.profiler.ProfilerActivity.CPU,
        #             torch.profiler.ProfilerActivity.CUDA],
        #         record_shapes=True,
        #         profile_memory=True,
        #         with_stack=False,
        #         ) as prof:
        #     self.train_loop()
        # # prof.export_chrome_trace("profiler_trace.json")
        # # writer.close()
        # print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
        
        # log training info
        if self.estop.epoch > 0:
            end = time.time()
            sec_per_step = (end - start) / self.estop.epoch
            self.info.update({'sec_per_step':sec_per_step})
            if 'total_lower_steps' in self.info and 'total_lower_time' in self.info and self.info['total_lower_steps']>0:
                self.info.update({'sec_per_lower_step':self.info['total_lower_time']/self.info['total_lower_steps']})
        
        # log memory usage
        mem =  get_mem_stats(self.device)
        self.info.update(mem)

        # log info
        self.logger.log_params(flatten(self.info))

    def save_optimizer(self):
        # save optimizer

        for key in self.optimizer.keys():
            fname = f"optimizer_{key}.pth"
            fpath = self.logger.gen_path(fname)
            torch.save(self.optimizer[key].state_dict(), fpath)
            print(f'save optimizer to {fpath}')
    

    def load_optim(self, optimizer, fpath):
        if not os.path.exists(fpath):
            print(f'optimizer file {fpath} not found, use default optimizer')
            return
        
        print(f'restore optimizer from {fpath}')
        state_dict = torch.load(fpath, map_location=self.device, weights_only=True)

        # Ad-hoc fix for loading a subset of optimizer parameters.
        # This handles the case where the saved optimizer trained more parameters (e.g., param_net + param_pde)
        # than the current optimizer (e.g., only param_net)
        
        saved_param_groups = state_dict['param_groups']
        current_param_groups = optimizer.param_groups

        # Check if number of groups matches
        if len(saved_param_groups) != len(current_param_groups):
            print(f"Warning: Mismatch in the number of optimizer parameter groups. "
                  f"Saved: {len(saved_param_groups)}, Current: {len(current_param_groups)}. "
                  "Skipping optimizer load.")
            return

        # For each group, adjust the number of parameters if needed
        needs_adjustment = False
        for saved_group, current_group in zip(saved_param_groups, current_param_groups):
            saved_param_ids = saved_group['params']
            current_params = current_group['params']
            
            # If the saved state has more parameters, truncate it
            if len(saved_param_ids) > len(current_params):
                needs_adjustment = True
                print(f"Warning: Optimizer parameter group has a size mismatch. "
                      f"Saved: {len(saved_param_ids)}, Current: {len(current_params)}. "
                      f"Loading state for the first {len(current_params)} parameters.")
                # Truncate the list of parameter IDs in the saved state_dict
                saved_group['params'] = saved_param_ids[:len(current_params)]
                
                # Also need to remove the extra parameter states from state_dict['state']
                for param_id in saved_param_ids[len(current_params):]:
                    if param_id in state_dict['state']:
                        del state_dict['state'][param_id]
        
        if needs_adjustment:
            print("Adjusted optimizer state_dict to match current parameter count")
        
        optimizer.load_state_dict(state_dict)

        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)

    
    def restore_optimizer(self, dirname):
        # restore optimizer, need dirname
        if self.opts['reset_optim']:
            print('do not restore optimizer, reset optimizer to default')
            return

        for key in self.optimizer.keys():
            fname = f"optimizer_{key}.pth"
            fpath = os.path.join(dirname, fname)
            self.load_optim(self.optimizer[key], fpath)
            

    def save_net(self):
        # save network
        
        net_path = self.logger.gen_path("net.pth")
        torch.save(self.net.state_dict(), net_path)
        print(f'save model to {net_path}')

    def restore_net(self, net_path):
        
        state_dict = torch.load(net_path,map_location=self.device, weights_only=True)

        # set strict to False, to allow loading partial model for loRA
        self.net.load_state_dict(state_dict, strict=False)

        # if net pde param is different from prob.init_param, need to update
        with torch.no_grad():
            # if net have all_params_dict attribute
            # Neural operators does not have all_params_dict attribute
            if hasattr(self.net, 'all_params_dict'):
                for key, val in self.net.all_params_dict.items():
                    if key in self.pde.opts['init_param']:
                        new_val = self.pde.opts['init_param'][key]
                        print(f'update pde parameter {key} from {val} to {new_val}')
                        tensor = torch.tensor([[new_val]], dtype=val.dtype, device=val.device)
                        self.net.all_params_dict[key].data = tensor

        print(f'restore model from {net_path}')

    def save_dataset(self):
        ''' make prediction and save dataset'''
        self.pde.make_prediction(self.net)

        # if self.pde has attribute names2save, save only those variables, mainly for GBMproblem
        dataset_path = self.logger.gen_path("dataset.mat")
        names2save = []
        if hasattr(self.pde, 'names2save'):
            names2save = self.pde.names2save
        self.pde.dataset.save(dataset_path, names2save=names2save)
        


    def save(self):
        '''saving dir from logger'''
        
        # save prediction
        self.save_dataset()

        # if max_iter is 0, do not save optimizer and net
        if self.opts['max_iter']>0:
            self.save_optimizer()
            self.save_net()
        

    def restore(self, dirname):
        # restore optimizer and network
        self.restore_optimizer(dirname)

        fnet = os.path.join(dirname, 'net.pth')
        self.restore_net(fnet)

    @abc.abstractmethod
    def train_loop(self):
        # training loop
        pass


class BiLevelTrainer(Trainer):

    def __init__(self, *args):
        super().__init__(*args)

        # optimizer and scheduler for lower level
        optimizer_net = optimizer_dictionary[self.opts['optim_net']]
        optim_net_opts = self.opts['opts_net']
        self.optimizer['param_net'] = optimizer_net(self.net.param_net, lr=self.opts['lr_net'], **optim_net_opts)

        scheduler_net = getattr(optim.lr_scheduler, self.opts['sch_net'])
        self.scheduler['param_net'] = scheduler_net(self.optimizer['param_net'], **self.opts['schopt_net'])

        # optimizer and scheduler for upper level
        optimizer_pde = optimizer_dictionary[self.opts['optim_pde']]
        optim_pde_opts = self.opts['opts_pde']
        self.optimizer['param_pde'] = optimizer_pde(self.net.param_pde_trainable, lr=self.opts['lr_pde'], **optim_pde_opts)
        
        # setup learning rate scheduler
        scheduler_pde = getattr(optim.lr_scheduler, self.opts['sch_pde'])
        self.scheduler['param_pde'] = scheduler_pde(self.optimizer['param_pde'], **self.opts['schopt_pde'])


    def train_loop(self):
        # single optimizer for all parameters, change learning rate manuall
        
    
        def get_lower_loss(yes_grad = True):
            # self.optimizer['param_net'].zero_grad()
            weighted_sum, weighted_loss_comp, unweighted_loss_comp = self.lossCollection.get_wloss_sum_comp(self.loss_net, yes_grad)
            return weighted_sum, weighted_loss_comp, unweighted_loss_comp
        
        def step_lower(loss_lower):
            self.set_grad(self.net.param_net, loss_lower)
            self.optimizer['param_net'].step()
            self.scheduler['param_net'].step()
        
        def get_upper_loss(yes_grad = True):
            # self.optimizer['param_pde'].zero_grad()
            # should always compute gradient, even for validation, because residual loss or l2grad need gradient
            weighted_sum, weighted_loss_comp, unweighted_loss_comp = self.lossCollection.get_wloss_sum_comp(self.loss_pde, yes_grad)
            return weighted_sum, weighted_loss_comp, unweighted_loss_comp

        def step_upper(loss_upper):
            self.set_grad(self.net.param_pde_trainable, loss_upper)
            self.optimizer['param_pde'].step()
            self.scheduler['param_pde'].step()
        
        epoch = 0

        w_upper = 0.0
        w_upper_comp = {}
        uw_upper_comp = {}
        
        w_upper, w_upper_comp, uw_upper_comp = get_upper_loss(True)
        
        # zero grad for all parameters
        self.optimizer['param_net'].zero_grad()
        self.optimizer['param_pde'].zero_grad()

        while True:
            at_lower = 0

            w_lower, w_lower_comp, uw_lower_comp = get_lower_loss(True)

            
            # check early stopping
            # if skip_upper, only monitor lower loss, otherwise monitor upper loss
            loss_to_monitor = w_upper
            stophere = self.estop(epoch, loss_to_monitor)

            # for component, log unweighted loss, for total, log weighted loss
            self.log_stat(epoch, stophere, **uw_lower_comp, **uw_upper_comp, lower=at_lower, lowertot=w_lower, uppertot=w_upper)
            self.validate(epoch, stophere)

            if stophere:
                self.log_stat(epoch, stophere, **uw_lower_comp, **uw_upper_comp, lower=at_lower, lowertot=w_lower, uppertot=w_upper)
                break  
                
            
            ### lower level
            epoch_lower = 0
            while w_lower > self.opts['tol_lower']:
                
                at_lower = 1

                self.acc_grad(self.net.param_net, w_lower)
                if epoch % self.opts['acc_iter'] == 0:
                    self.optimizer['param_net'].step()
                    self.scheduler['param_net'].step()
                    self.optimizer['param_net'].zero_grad()

                epoch_lower += 1
                epoch += 1

                w_lower, w_lower_comp, uw_lower_comp = get_lower_loss(True)
                w_upper, w_upper_comp, uw_upper_comp = get_upper_loss(True)

                self.log_stat(epoch, stophere, **uw_lower_comp, **uw_upper_comp, lower=at_lower, lowertot=w_lower, uppertot=w_upper)
                self.validate(epoch, stophere)

                # get next batch
                self.pde.dataset.next_batch()

                if epoch_lower == self.opts['max_iter_lower']:
                    break
            ### end of lower level

            at_lower = 0

            
            if self.opts['simu_update']:
                # simulaneous update
                # need to compute upper before stepping lower
                w_upper, w_upper_comp, uw_upper_comp = get_upper_loss(True)
                self.acc_grad(self.net.param_pde_trainable, w_upper)
                # step lower
                self.acc_grad(self.net.param_net, w_lower)

                # when to step accumulate gradient
                if epoch % self.opts['acc_iter'] == 0:
                    self.optimizer['param_net'].step()
                    self.scheduler['param_net'].step()
                    self.optimizer['param_net'].zero_grad()
                    # then step upper
                    self.optimizer['param_pde'].step()
                    self.scheduler['param_pde'].step()
                    self.optimizer['param_pde'].zero_grad()
            else:
                # rase error
                raise NotImplementedError('alternating update not implemented')
            
            self.pde.dataset.next_batch()
            epoch += 1
            # next cycle


class SingleLevelTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # name for optimizer and scheduler
        self.group_name = None
        # list of loss to monitor
        self.which_loss = None

    def train_loop(self):
        '''
        vanilla training of network, update all parameters simultaneously
        '''
        epoch = 0
        
        group_name = self.group_name

        while True:

            self.optimizer[group_name].zero_grad()
            
            wtotal, wloss_comp, uwloss_comp = self.lossCollection.get_wloss_sum_comp(self.which_loss, True)

            # if self.loss_monitor is non-empty, monitor the loss
            if self.loss_monitor:
                wmonitor = 0.0
                for key in self.loss_monitor:
                    wmonitor += wloss_comp[key]
            else:
                wmonitor = wtotal

            # check early stopping
            stophere = self.estop( epoch, wmonitor)

            # print statistics at interval or at stop
            self.log_stat(epoch, stophere, **uwloss_comp, total=wtotal)
            self.validate(epoch, stophere)

            if stophere:
                break  

            # take gradient of residual loss w.r.t all parameters
            # do not use setgrad here. if vanilla init, pde_param requires grad = False
            wtotal.backward()
            self.optimizer[group_name].step()
            self.scheduler[group_name].step()
            # next cycle
            self.pde.dataset.next_batch()
            
            epoch += 1


class PinnTrainer(SingleLevelTrainer):
    
    def __init__(self, *args):
        super().__init__(*args)

        optimizer = optimizer_dictionary[self.opts['optim_net']]
        optim_options = self.opts['opts_net']

        # Use parameter groups for different learning rates
        param_groups = [
            {'params': self.net.param_net, 'lr': self.opts['lr_net']},
            {'params': self.net.param_pde, 'lr': self.opts['lr_pde']}
        ]
        self.optimizer['param_net'] = optimizer(param_groups, **optim_options)
        
        scheduler_net = getattr(optim.lr_scheduler, self.opts['sch_net'])
        self.scheduler['param_net'] = scheduler_net(self.optimizer['param_net'], **self.opts['schopt_net'])
        
        self.which_loss = self.opts['loss_net']
        self.group_name = 'param_net'

class OperatorPretrainTrainer(SingleLevelTrainer):
    
    def __init__(self, *args):
        super().__init__(*args)

        optimizer = optimizer_dictionary[self.opts['optim']]
        optim_options = self.opts['opts']

        # param_all include all parameters, including requires_grad = False (some pde parameter and embedding)
        self.optimizer['param_net'] = optimizer(self.net.parameters(), lr=self.opts['lr'], **optim_options)
        
        scheduler_net = getattr(optim.lr_scheduler, self.opts['sch'])
        self.scheduler['param_net'] = scheduler_net(self.optimizer['param_net'], **self.opts['schopt'])
        
        self.which_loss = self.opts['loss_net']
        self.group_name = 'param_net'

class OperatorInverseTrainer(SingleLevelTrainer):
    
    def __init__(self, *args):
        super().__init__(*args)

        # set all parameter untrainable
        for param in self.net.parameters():
            param.requires_grad = False
        # set pde_param trainable
        for param in self.net.pde_params_dict:
            self.net.pde_params_dict[param].requires_grad = True

        trainable_param = [self.net.pde_params_dict[key] for key in self.net.pde_params_dict]

        optimizer = optimizer_dictionary[self.opts['optim']]
        optim_options = self.opts['opts']
        self.optimizer['param_net'] = optimizer(trainable_param, lr=self.opts['lr'], **optim_options)
        
        scheduler_net = getattr(optim.lr_scheduler, self.opts['sch'])
        self.scheduler['param_net'] = scheduler_net(self.optimizer['param_net'], **self.opts['schopt'])
        
        self.which_loss = self.opts['loss_net']
        self.group_name = 'param_net'

        
# class BiloInitTrainer(SingleLevelTrainer):
#     def __init__(self, *args):
#         super().__init__(*args)
        
#         optimizer = optimizer_dictionary[self.opts['optim_net']]
#         optim_options = self.opts['opts_net']
#         # param_all include all parameters, including requires_grad = False (some pde parameter and embedding)
#         trainable_param = self.net.param_net
        
#         # for initialization. 
#         # 1. Scalar case, param not trainable
#         # 2. Function case, param trainable
#         # wanring: not handling cases inferring both scalar and function
#         if self.net.with_func:
#             trainable_param += self.net.param_pde

#         self.optimizer['param_net'] = optimizer(trainable_param, lr=self.opts['lr_net'], **optim_options)
        
#         scheduler_net = getattr(optim.lr_scheduler, self.opts['sch_net'])
#         self.scheduler['param_net'] = scheduler_net(self.optimizer['param_net'], **self.opts['schopt_net'])

#         self.which_loss = self.opts['loss_net']
#         self.group_name = 'param_net'

class BiloInitTrainer(BiLevelTrainer):
    def __init__(self, *args):
        super().__init__(*args)

class UpperTrainer(SingleLevelTrainer):
    def __init__(self, *args):
        super().__init__(*args)

        optimizer = optimizer_dictionary[self.opts['optim_pde']]
        optim_options = self.opts['opts_pde']
        
        self.optimizer['param_pde'] = optimizer(self.net.param_pde_trainable, lr=self.opts['lr_pde'], **optim_options)
        scheduler_pde = getattr(optim.lr_scheduler, self.opts['sch_pde'])
        self.scheduler['param_pde'] = scheduler_pde(self.optimizer['param_pde'], **self.opts['schopt_pde'])
        
        self.which_loss = self.opts['loss_pde']
        self.group_name = 'param_pde'
