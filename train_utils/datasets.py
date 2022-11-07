import scipy.io
import numpy as np

try:
    from pyDOE import lhs
    # Only needed for PINN's dataset
except ImportError:
    lhs = None

import torch
from torch.utils.data import Dataset
from .utils import get_grid3d, convert_ic


def online_loader(sampler, S, T, time_scale, batchsize=1):
    while True:
        u0 = sampler.sample(batchsize)
        a = convert_ic(u0, batchsize,
                       S, T,
                       time_scale=time_scale)
        yield a


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch

class NSLoader(object):
    def __init__(self, datapath1,
                 nx, nt,
                 datapath2=None, sub=1, sub_t=1,
                 N=100, t_interval=1.0):
        '''
        Load data from npy and reshape to (N, X, Y, T, 2)
        Args:
            datapath1: path to data
            nx:
            nt:
            datapath2: path to second part of data, default None
            sub:
            sub_t:
            N:
            t_interval:
        '''
        self.S = nx // sub
        self.T = int(nt * t_interval) // sub_t + 1
        self.time_scale = t_interval
        data1 = np.load(datapath1)
        data1 = torch.tensor(data1, dtype=torch.float)[..., ::sub_t, ::sub, ::sub, ::]

        if t_interval == 0.5:
            data1 = self.extract(data1)
            if datapath2 is not None:
                data2 = self.extract(data2)
        part1 = data1.permute(0, 2, 3, 1, 4)
        self.data = part1

    def make_loader(self, n_sample, batch_size, start=0, train=True):
        
        if train:
            a_data = self.data[start:start + n_sample, :, :, 0, :].reshape(n_sample, self.S, self.S, 2)
            u_data = self.data[start:start + n_sample].reshape(n_sample, self.S, self.S, self.T, 2)
        else:
            a_data = self.data[-n_sample:, :, :, 0, :].reshape(n_sample, self.S, self.S, 2)
            u_data = self.data[-n_sample:].reshape(n_sample, self.S, self.S, self.T, 2)
        
        a_data = a_data.reshape(n_sample, self.S, self.S, 1, 2).repeat([1, 1, 1, self.T, 1])
        
        gridx, gridy, gridt = get_grid3d(self.S, self.T, time_scale=self.time_scale)
        
        a_data = torch.cat((gridx.repeat([n_sample, 1, 1, 1, 1]), gridy.repeat([n_sample, 1, 1, 1, 1]),
                            gridt.repeat([n_sample, 1, 1, 1, 1]), a_data), dim=-1)
        
        dataset = torch.utils.data.TensorDataset(a_data, u_data)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=train)
        return loader

    def make_dataset(self, n_sample, start=0, train=True):
        
        if train:
            a_data = self.data[start:start + n_sample, :, :, 0, :].reshape(n_sample, self.S, self.S, 2)
            u_data = self.data[start:start + n_sample].reshape(n_sample, self.S, self.S, self.T, 2)
        else:
            a_data = self.data[-n_sample:, :, :, 0, :].reshape(n_sample, self.S, self.S, 2)
            u_data = self.data[-n_sample:].reshape(n_sample, self.S, self.S, self.T, 2)
        
        a_data = a_data.reshape(n_sample, self.S, self.S, 1, 2).repeat([1, 1, 1, self.T, 1])
        gridx, gridy, gridt = get_grid3d(self.S, self.T)
        
        a_data = torch.cat((
            gridx.repeat([n_sample, 1, 1, 1, 1]),
            gridy.repeat([n_sample, 1, 1, 1, 1]),
            gridt.repeat([n_sample, 1, 1, 1, 1]),
            a_data), dim=-1)
        
        dataset = torch.utils.data.TensorDataset(a_data, u_data)
        return dataset

    @staticmethod
    def extract(data):
        '''
        Extract data with time range 0-0.5, 0.25-0.75, 0.5-1.0, 0.75-1.25,...
        Args:
            data: tensor with size N x 129 x 128 x 128 x 2

        Returns:
            output: (4*N-1) x 65 x 128 x 128 x 2
        '''
        T = data.shape[1] // 2
        interval = data.shape[1] // 4
        N = data.shape[0]
        new_data = torch.zeros(4 * N - 1, T + 1, data.shape[2], data.shape[3], 2)
        for i in range(N):
            for j in range(4):
                if i == N - 1 and j == 3:
                    # reach boundary
                    break
                if j != 3:
                    new_data[i * 4 + j] = data[i, interval * j:interval * j + T + 1, :]
                else:
                    new_data[i * 4 + j, 0: interval] = data[i, interval * j:interval * j + interval, :]
                    new_data[i * 4 + j, interval: T + 1] = data[i + 1, 0:interval + 1, :]
        return new_data