# base class for PDE problem
# A problem shoudl include dataset and network
import os
from abc import ABC, abstractmethod

import torch
import matplotlib.pyplot as plt

from util import generate_grf, to_double, add_noise, mse, error_logging_decorator
from MatDataset import MatDataset
from DenseNet import DenseNet

class BaseProblem(ABC):
    def __init__(self, **kwargs):
        super().__init__()
        self.dataset = None

        # dimension of PDE input and output
        self.input_dim = None
        self.output_dim = None
        # transformation of network
        self.lambda_transform = None
        self.opts = {}
        self.tag = []
        
        # collection of loss functions
        self.loss_dict = {'res': self.resloss,
        'fullresgrad': self.fullresgradloss,
        'fdmresgrad': self.fdmresgrad,
        'funcloss':self.func_mse,
        'data': self.get_data_loss,
        'bc': self.bcloss}


        # all parameters in the PDE, including those not used in the network, a dict to initialize network
        self.all_params_dict = {}
        # list of PDE parameters as input to BILO, might differ from all_params_dict. 
        # e.g. imaging thresholds are not input to BILO
        # if none, same as all_params_dict
        self.pde_params = []

    @abstractmethod
    def residual(self, nn, x):
        pass
    
    # take network, return residual and predictio
    def get_res_pred(self, net):
        ''' get residual and prediction'''
        self.res, self.pred = self.residual(net, self.dataset['X_res_train'])
        return self.res, self.pred

    def get_data_loss(self, net):
        # get data loss
        u_pred = net(self.dataset['X_dat_train'], pde_params_dict=net.pde_params_dict)
        loss = torch.mean(torch.square(u_pred - self.dataset['u_dat_train']))

        return loss
    
    def func_mse(self, net):
        # problem with unkonwn function need to implement this loss
        raise NotImplementedError

    def bcloss(self, net):
        # problem with unkonwn function need to implement this loss
        raise NotImplementedError

    def resloss(self, net):
        self.res, self.upred_res = self.get_res_pred(net)
        val_loss_res = torch.mean(torch.square(self.res))
        return val_loss_res

    def fullresgradloss(self, net):
        # compute gradient of residual w.r.t. parameter on every residual point.
        self.res_unbind = self.res.unbind(dim=1) # unbind residual into a list of 1d tensor

        n = self.res.shape[0]
        resgradmse = 0.0        
        for pname in net.pde_params_dict:
            if pname not in net.trainable_param:
                continue
            for j in range(self.output_dim):
                tmp = torch.autograd.grad(self.res_unbind[j], net.params_expand[pname], grad_outputs=torch.ones_like(self.res_unbind[j]),
                create_graph=True, retain_graph=True,allow_unused=True)[0]
                resgradmse += torch.sum(torch.pow(tmp, 2))
        
        return resgradmse/n

    def fdmresgrad(self, net):
        # FDM version of fullresgradloss

        n = self.res.shape[0]
        delta = 0.01
        resgradmse = 0.0        

        # self.res_unbind = self.res.unbind(dim=1)

        # make copy of pde_params_dict and detach()
        original_pde_params = {k:v for k,v in net.pde_params_dict.items()}
        

        for pname in net.pde_params_dict:
            if pname not in net.trainable_param:
                continue
            for j in range(self.output_dim):
                
                # # central diff, note: fwd diff does not work
                #compute resiudal at pname + delta
                net.pde_params_dict[pname] = original_pde_params[pname].detach().clone() + delta
                res_pos, _ = self.get_res_pred(net)
                #compute resiudal at pname - delta
                net.pde_params_dict[pname] = original_pde_params[pname].detach().clone() - delta
                res_neg, _ = self.get_res_pred(net)
                # restore original value
                tmp = (res_pos[:,j] - res_neg[:,j])/(2*delta)

                resgradmse += torch.sum(torch.pow(tmp, 2))

        net.pde_params_dict = original_pde_params

        return resgradmse/n
    
    def config_traintype(self, traintype):
        pass
    
    # compute validation statistics
    @torch.no_grad()
    def validate(self, nn):
        '''output validation statistics'''
        v_dict = {}
        for vname in nn.trainable_param:
            err = torch.abs(nn.all_params_dict[vname] - self.gt_param[vname])
            v_dict[f'abserr_{vname}'] = err
            v_dict[vname] = nn.all_params_dict[vname]    
        return v_dict
    
    @torch.no_grad()
    def make_prediction(self, net):
        # make prediction at original X_dat and X_res
        self.dataset['upred_res_test'] = net(self.dataset['X_res_test'], net.pde_params_dict)
        self.dataset['upred_dat_test'] = net(self.dataset['X_dat_test'], net.pde_params_dict)
        if hasattr(self, 'u_exact'):
            self.dataset['uinf_dat_test'] = self.u_exact(self.dataset['X_dat_test'], net.pde_params_dict)
        self.prediction_variation(net)

    def setup_network(self, **kwargs):
        '''setup network, get network structure if restore'''
        # first copy self.pde.param, which include all pde-param in network
        # then update by init_param if provided
        kwargs['input_dim'] = self.input_dim
        kwargs['output_dim'] = self.output_dim

        all_param = self.all_params_dict.copy()
        init_param = self.opts['init_param']
        if init_param is not None:
            all_param.update(init_param)

        net = DenseNet(**kwargs,
                        lambda_transform = self.lambda_transform,
                        all_params_dict = all_param,
                        pde_params = self.pde_params,
                        trainable_param = self.opts['trainable_param'])
        return net


    @torch.no_grad()
    def prediction_variation(self, net, list_params:list[str]=None):
        # make prediction with different parameters
        # variation name = f'var_{param_name}_{delta_i}_pred'

        if net.with_param == False:
            print('no BiLO, skip prediction variation')
            return

        if 'X_dat_test' in self.dataset:
            x_test = self.dataset['X_dat_test']
        elif 'x_dat' in self.dataset:
            x_test = self.dataset['x_dat']
        elif 'X_dat' in self.dataset:
            x_test = self.dataset['X_dat']
        else:
            raise ValueError('X_dat_test or x_dat not found in dataset')
        # percentage of variation
        deltas = [0.0, 0.1, -0.1, 0.2, -0.2, 0.5, -0.5]
        self.dataset['deltas'] = deltas
        
        # go through all the trainable pde parameters
        if list_params is not None:
            params_to_vary = list_params
        else:
            params_to_vary = [p for p in net.pde_params_dict]
            # only pde param has variation
        
        if len(params_to_vary) == 0:
            print('no parameter to vary, skip prediction variation')
            return

        for k in params_to_vary:
            # copy the parameters, DO NOT modify the original parameters
            # need to be in the loop, reset each time
            # tmp_param_dict = {k: v.clone() for k, v in net.pde_params_dict.items()}
            tmp_param_dict = {}
            for k_, v_ in net.pde_params_dict.items():
                if isinstance(v_, torch.Tensor):
                    tmp_param_dict[k_] = v_.clone()
                elif isinstance(v_, torch.nn.Module):
                    tmp_param_dict[k_] = v_(x_test)
                else:
                    raise ValueError('unknown type of pde parameter')
                
            # original value
            param_value = tmp_param_dict[k]
            param_name = k

            for delta_i, delta in enumerate(deltas):
                new_value = param_value * (1 + delta)
                
                tmp_param_dict[param_name] = new_value
                # to(x_test.device)

                u_test = net(x_test, tmp_param_dict)
                vname = f'var_{param_name}_{delta_i}_pred'
                self.dataset[vname] = u_test

                if hasattr(self, 'u_exact'):
                    u_exact = self.u_exact(x_test, tmp_param_dict)
                    vname = f'var_{param_name}_{delta_i}_exact'
                    self.dataset[vname] = u_exact

    @abstractmethod
    def setup_dataset(self, dsopt, noise_opt):
        pass
    
    @error_logging_decorator
    def plot_variation(self, savedir=None):
        # plot variation of net w.r.t each parameter
        if 'deltas' not in self.dataset:
            print('no deltas found in dataset, skip plotting variation')
            return

        if 'X_dat_test' in self.dataset:
            x_test = self.dataset['X_dat_test']
        elif 'x_dat' in self.dataset:
            x_test = self.dataset['x_dat']
        else:
            raise ValueError('X_dat_test or x_dat not found in dataset')

        deltas = self.dataset['deltas']

        var_names = self.dataset.filter('var_')
        # find unique parameter names
        param_names = list(set([v.split('_')[1] for v in var_names]))

        # for each varname, plot the solution and variation
        for varname in param_names:
            fig, ax = plt.subplots()

            # for each delta
            for i_delta,delta in enumerate(deltas):

                vname_pred = f'var_{varname}_{i_delta}_pred'
                # plot prediction
                u_pred = self.dataset[vname_pred]
                ax.plot(x_test, u_pred, label=f'NN $\Delta${varname} = {delta:.2f}')

                # plot exact if available
                if hasattr(self, 'u_exact'):
                    vname_exact = f'var_{varname}_{i_delta}_exact'
                    color = ax.lines[-1].get_color()
                    u_exact = self.dataset[vname_exact]
                    ax.plot(x_test, u_exact, label=f'exact $\Delta${varname} = {delta:.2f}',color=color,linestyle='--')

            ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

            if savedir is not None:
                fname = f'fig_var_{varname}.png'
                fpath = os.path.join(savedir, fname)
                fig.savefig(fpath, dpi=300, bbox_inches='tight')
                print(f'fig saved to {fpath}')

        return

    @error_logging_decorator
    def plot_prediction(self,savedir=None, vname='u'):
        ''' plot prediction at x_dat_train
        '''
        x_dat_train = self.dataset['X_dat_train']
        x_dat_test = self.dataset['X_dat_test']

        # visualize the results
        fig, ax = plt.subplots()

        # assume single dimension output

        # plot prediction
        v = f'{vname}pred_dat_test'
        if v in self.dataset:
            ax.plot(x_dat_test, self.dataset[v], label=f'pred {vname}')
        
        # plot test data
        v = f'{vname}_dat_test'
        if v in self.dataset:
            ax.plot(x_dat_test, self.dataset[v], label=f'test {vname}', linestyle='--', color='black')
        
        # plot train data
        v = f'{vname}_dat_train'
        if v in self.dataset:
            ax.scatter(x_dat_train, self.dataset[v], label=f'train {vname}', marker='.')
        
        # plot solution with inferred parameters
        if f'{vname}inf_dat_test' in self.dataset:
            ax.plot(x_dat_test, self.dataset[f'{vname}inf_dat_test'], label=f'inf {vname}', linestyle=':')

        ax.legend()

        if savedir is not None:
            fpath = os.path.join(savedir, f'fig_pred_{vname}.png')
            fig.savefig(fpath, dpi=300, bbox_inches='tight')
            print(f'fig saved to {fpath}')

        return fig, ax

    def visualize(self, savedir=None):
        # visualize the results
        self.dataset.to_np()
        self.plot_prediction(savedir=savedir)
        self.plot_variation(savedir=savedir)
    
    
    
