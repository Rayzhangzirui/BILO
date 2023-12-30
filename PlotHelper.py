#!/usr/bin/env python
import sys
import torch
import torch.nn as nn
from config import *
from util import *

from DensePoisson import *
from MlflowHelper import MlflowHelper
from DataSet import DataSet
from Problems import *
from matplotlib import pyplot as plt

DEBUG = True
def catch_plot_errors(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if DEBUG:
                raise
            else:
                print(f"An error occurred while plotting: {e}")
    return wrapper

class PlotHelper:
    def __init__(self, pde, dataset, **kwargs) -> None:

        
        self.pde = pde
        self.dataset = dataset

        # default options
        self.opts = {}
        self.opts['yessave'] = False
        self.opts['save_dir'] = './tmp'
        self.opts.update(kwargs)

    @catch_plot_errors
    def plot_variation(self, net):
        # plot variation of net w.r.t each parameter

        x_test = self.dataset['x_res_test']

        u_test = net(x_test)
        u_init_test = self.pde.u_exact(x_test, net.params_dict)

        fig, ax = plt.subplots()

        # get device of current network
        device = next(net.parameters()).device
        
        # for each net.params_dict, plot the solution and variation
        for k, v in net.params_dict.items():
            param_value = net.params_dict[k].item()
            param_name = k

            deltas = [0.0, 0.1, -0.1]
            for delta in deltas:    
                
                # replace parameter
                with torch.no_grad():    
                    new_value = param_value + delta
                    net.params_dict[param_name].data = torch.tensor([[new_value]]).to(device)
                
                u_test = net(x_test)
                ax.plot(x_test.cpu().numpy(), u_test.cpu().detach().numpy(), label=f'NN {param_name} = {new_value:.2e}')

                u_exact_test = self.pde.u_exact(x_test, net.params_dict)
                # get the color of previous line
                color = ax.lines[-1].get_color()
                ax.plot(x_test.cpu().numpy(), u_exact_test.cpu().detach().numpy(), label=f'exact {param_name} = {new_value:.2e}',color=color,linestyle='--')
            # set net.D
            ax.legend(loc="best")

            if self.opts['yessave']:
                self.save(f'fig_variation_{param_name}.png', fig)

        return fig, ax

    
    def save(self, fname, fig):
        # save current figure
        fpath = os.path.join(self.opts['save_dir'], fname)
        fig.savefig(fpath, dpi=300, bbox_inches='tight')
        print(f'{fname} saved to {fpath}')
     
    @catch_plot_errors
    def plot_prediction2(self, net, dataset):
        x_dat_test = self.dataset['x_dat_test']
        u_dat_test = self.dataset['u_dat_test']

        with torch.no_grad():
            upred = net(x_dat_test)
        
        # move to cpu
        x_dat_test = x_dat_test.cpu().detach().numpy()
        upred = upred.cpu().detach().numpy()
        u_test = u_dat_test.cpu().detach().numpy()

        # visualize the results
        fig, ax = plt.subplots()
        
        # plot nn prediction
        ax.plot(x_dat_test, upred, label='pred')
        # plot exact solution
        ax.plot(x_dat_test, u_test, label='test',linestyle='--')

        if self.opts['yessave']:
                self.save(f'fig_pred_xtest.png', fig)

        return fig, ax


    @catch_plot_errors
    def plot_prediction(self, net):
        x_test = self.dataset['x_res_test']

        u_test = net(x_test)
        u_init_test = self.pde.u_exact(x_test, self.pde.init_param)
        u_exact_test = self.pde.u_exact(x_test, self.pde.exact_param)

        # excat u with predicted D, for comparison
        u_exact_pred_D_test = self.pde.u_exact(x_test, net.params_dict)

        x_res_train = self.dataset['x_res_train']
        u_pred = net(x_res_train)
        u_res = self.dataset['u_res_train']

        # move to cpu
        x_test = x_test.cpu().detach().numpy()
        u_test = u_test.cpu().detach().numpy()
        u_init_test = u_init_test.cpu().detach().numpy()
        u_exact_test = u_exact_test.cpu().detach().numpy()
        u_exact_pred_D_test = u_exact_pred_D_test.cpu().detach().numpy()
        x_res_train = x_res_train.cpu().numpy()
        u_pred = u_pred.cpu().detach().numpy()
        u_res = u_res.cpu().detach().numpy()

        # visualize the results
        fig, ax = plt.subplots()
        
        # plot nn prediction
        ax.plot(x_test, u_test, label='nn pred')
        # plot exact solution
        ax.plot(x_test, u_exact_test, label='GT')
        ax.plot(x_test, u_exact_pred_D_test, label='exact pred D',linestyle='--',color='gray')
        
        # scatter plot of training data
        ax.plot(x_res_train, u_pred, '.', label='train-pred')
        ax.plot(x_res_train, u_res, '.', label='train-data')


        ax.legend(loc="upper right")
        # add title

        D = net.params_dict['D'].item()
        ax.set_title(f'final D: {D:.3f}')

        if self.opts['yessave']:
            self.save('fig_pred.png', fig)

        return fig, ax
    
    @catch_plot_errors
    def plot_loss(self, hist, loss_names=None):
        # plot loss history
        fig, ax = plt.subplots()
        x = hist['steps']

        if loss_names is None:
            loss_names = list(hist.keys())
            # remove step
            loss_names.remove('steps')

        for lname in loss_names:
            if lname in hist:
                ax.plot(x, hist[lname], label=lname)
            else:
                print(f'{lname} not in hist')
        
        ax.set_yscale('log')
        
        ax.legend(loc="upper right")
        ax.set_title('Loss history')

        if self.opts['yessave']:
            self.save('fig_loss.png', fig)

        return fig, ax


def output_svd(m1, m2, layer_name):
    # take two models and a layer name, output the svd of the weight and svd of the difference
    W1 = m1.state_dict()[layer_name]
    W2 = m2.state_dict()[layer_name]
    _, s1, _ = torch.svd(W1)
    _, s2, _ = torch.svd(W2)
    _, s_diff, _ = torch.svd(W1 - W2)
    return s1, s2, s_diff

@catch_plot_errors
def plot_svd(s1, s2, s_diff, name1, name2, namediff):
    # sve plot, 
    # s1, s2, s_diff are the svd of the weight and svd of the difference
    # name1, name2 are the names of the two models
    # layer_name is the name of the layer

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(s1, label=name1)
    ax.plot(s2, label=name2)
    ax.plot(s_diff, label=namediff)
    ax.set_yscale('log')
    return fig, ax
    



if __name__ == "__main__":
    # visualize the results for single run
    
    exp_name = sys.argv[1]
    run_name = sys.argv[2]

    # get run id from mlflow, load hist and options
    helper = MlflowHelper()
    run_id = helper.get_id_by_name(exp_name, run_name)
    atf_dict = helper.get_active_artifact_paths(run_id)
    hist = helper.get_metric_history(run_id)
    opts = read_json(atf_dict['options.json'])

    # reecrate pde
    pde = create_pde_problem(**opts['pde_opts'])
    
    # load net
    nn = DensePoisson(**opts['nn_opts'])
    nn.load_state_dict(torch.load(atf_dict['net.pth']))

    # load dataset
    dataset = DataSet()
    dataset.readmat(atf_dict['dataset.mat'])


    ph = PlotHelper(pde, dataset, yessave=True, save_dir=atf_dict['artifacts_dir'])

    ph.plot_prediction(nn)
    D = nn.D.item()
    ph.plot_variation(nn)

    hist,_ = helper.get_metric_history(run_id)
    ph.plot_loss(hist,list(opts['weights'].keys())+['total'])
