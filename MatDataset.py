#!/usr/bin/env python
import os
import sys
from scipy.io import loadmat, savemat
import numpy as np
import torch
from itertools import cycle
# torhch dataset
from torch.utils.data import Dataset, DataLoader

class MatDataset(dict):
    '''data set class
    interface between .mat file and python
    access data set as dictionary
    '''
    def __init__(self, *args, **kwargs):
        # if only one argument, assume it is a file path, remove 
        if len(args) == 1:
            file_path = args[0]
            # check file exist
            if not os.path.exists(file_path):
                raise ValueError(f'File {file_path} not exist!')
            self.readmat(file_path, **kwargs)
        else:
            super().__init__(*args, **kwargs)

        self.loader = {}
        self.iter = {}
        self.batch = {}
        self.device = torch.device('cpu')


    def readmat(self, file_path, as_torch=True, ignore=[]):
        # load data from .mat file, skip meta data
        # default as torch tensor
        # ignore: list of variable names to ignore
        data = loadmat(file_path, mat_dtype=True)

        for key, value in data.items():
            
            if key.startswith("__"):
                # skip meta data
                continue
            if key in ignore:
                # skip ignored variables
                continue
            if isinstance(value,np.ndarray):

                if value.size == 1:
                    # if singleton, get number
                    value = value.item()
                else:
                    # squeeze array
                    value = np.squeeze(value)
                
                self[key] = value
        
        # otherwise it is a numpy array
        if as_torch:
            self.to_torch()
    
    def printsummary(self):
        '''print data set
        '''
        # print name, type, and shape of each variable
        for key, value in self.items():
            shape = None
            typename = type(value).__name__
            if isinstance(value, np.ndarray) or isinstance(value, torch.Tensor):
                shape = value.shape
                
            print(f'{key}:\t{typename}\t{shape}')
    
    def __str__(self):
        # return variable, type, and shape as string
        string = ''
        for key, value in self.items():
            shape = None
            typename = type(value).__name__ 
            if isinstance(value, np.ndarray) or isinstance(value, torch.Tensor):
                shape = value.shape
                
            string += f'{key}:\t{typename}\t{shape}\n'
        return string
    
    def save(self, file_path, names2save:list=[]):
        '''save data set to .mat file
        '''
        # save data set to .mat file
        self.to_np()
        if names2save == []:
            # save all variables
            print(f'save dataset to {file_path}')
            savemat(file_path, self)
        else:
            # only save specified variables
            print(f'save {names2save} to {file_path}')
            data2save = {name: self[name] for name in names2save if name in self}
            savemat(file_path, data2save)
    
    def to_torch(self):
        '''convert numpy array to torch tensor
        skip string
        '''
        print('convert dataset to torch')
        for key, value in self.items():
            if isinstance(value, np.ndarray):
                self[key] = torch.tensor(value,dtype=torch.float32)
    
    def to_np(self, d = None):
        '''convert tensor to cpu
        '''
        if d is None:
            d = self
            print('move dataset to cpu')

        for key, value in d.items():
            if isinstance(value, torch.Tensor):
                d[key] = value.cpu().detach().numpy()
            # if dictionary, recursively convert
            elif isinstance(value, dict):
                self.to_np(d = value)
            # if list, convert each element
            elif isinstance(value, list):
                for i in range(len(value)):
                    if isinstance(value[i], torch.Tensor):
                        value[i] = value[i].cpu().detach().numpy()
                
    
    def to_device(self, device):
        print(f'move dataset to {device}')
        self.to_torch()
        self.device = device
        for key, value in self.items():
            # if value is list, move each element
            if isinstance(value, list):
                for i in range(len(value)):
                    # if it's a tensor
                    if isinstance(value[i], torch.Tensor):
                        value[i] = value[i].to(device)
            else:
                if isinstance(value, torch.Tensor):
                    self[key] = value.to(device)
    
    def subsample_evenly_astrain(self, n, vars, suffix="_train", replace='', exclude_bd=False):
        '''uniformly downsample data set in the first dimension
        n is final number of samples
        '''
        N = self[vars[0]].shape[0]
        # step = (N-1)//(n-1)
        # interger linespace
        if exclude_bd:
            idx = np.linspace(0, N-1, n+2, dtype=int)
            idx = idx[1:-1]
        else:
            idx = np.linspace(0, N-1, n, dtype=int)

        for var in vars:
            if replace:
                new_name = var.replace(replace, suffix)
            else:
                new_name = var + suffix
            v = self[var][idx]

            self[new_name] = v.reshape(-1, 1)

    
    def filter(self, substr):
        ''' return list of key that contains substr
        '''
        return [key for key in self.keys() if substr in key]
    
    def subsample_firstn_astrain(self, n, vars):
        '''subsample first n row for training. 
        add _train suffix to variable name
        '''
        for var in vars:
            self[var+'_train'] = self[var][:n]
        
        new_vars = [var+'_train' for var in vars]
        return new_vars

    def subsample_unif_astrain(self, n, vars):
        '''subsample n row uniformly for training. 
        add _train suffix to variable name
        '''
        N = self[vars[0]].shape[0]
        idx = np.random.choice(N, n, replace=False)
        # keep order
        idx = np.sort(idx)
        new_vars = []
        for var in vars:
            new_var  = var+'_train'
            new_vars.append(new_var)
            self[new_var] = self[var][idx]
        return new_vars
            
    def remove(self, keys):
        '''remove variables that contains substr
        '''
        for key in keys:
            self.pop(key, None)
            print(f'remove {key}')

    def configure_dataloader(self, loader_name, loader_var, **loader_opts):
        self.loader[loader_name] = DataLoader(CustomDataset(self, loader_var), **loader_opts)
        self.iter[loader_name] = cycle(self.loader[loader_name])
        # initialize batch
        self.batch[loader_name] = next(self.iter[loader_name])
    
    def next_batch(self):
        # update batch for each loader
        for loader_name in self.loader:
            self.batch[loader_name] = next(self.iter[loader_name])
    


# pytorch dataset
class CustomDataset(Dataset):
    def __init__(self, matdataset, keys):
        self.matdataset = matdataset
        self.keys = keys
        # assert all keys have the same length
        assert all(self.matdataset[key].shape[0] == self.matdataset[keys[0]].shape[0] for key in keys)
        self.num_samples = self.matdataset[keys[0]].shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Create a dictionary to return all requested variables with their corresponding keys
        data = {key: self.matdataset[key][idx] for key in self.keys}
        return data

if __name__ == "__main__":
    # read mat file and print dataset
    filename  = sys.argv[1]

    # which variables to print
    vars2print = sys.argv[2:] if len(sys.argv) > 2 else None

    dataset = MatDataset(filename)
    
    dataset.to_torch()
    if vars2print is None:
        dataset.printsummary()
    else:
        for var in vars2print:
            print(dataset[var])
    