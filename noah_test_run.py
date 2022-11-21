print('Hello World')

import yaml
from argparse import ArgumentParser
import math
import torch
import numpy as np

print('Finished Importing Libaries')

def vor2vel(w, L=2 * np.pi):
    '''
    Convert vorticity into velocity
    Args:
        w: vorticity with shape (batchsize, num_x, num_y, num_t)

    Returns:
        ux, uy with the same shape
    '''
    batchsize = w.size(0)
    nx = w.size(1)
    ny = w.size(2)
    nt = w.size(3)
    device = w.device
    w = w.reshape(batchsize, nx, ny, nt)

    w_h = torch.fft.fft2(w, dim=[1, 2])
    # Wavenumbers in y-direction
    k_max = nx // 2
    N = nx
    k_x = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                     torch.arange(start=-k_max, end=0, step=1, device=device)), 0) \
        .reshape(N, 1).repeat(1, N).reshape(1, N, N, 1)
    k_y = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                     torch.arange(start=-k_max, end=0, step=1, device=device)), 0) \
        .reshape(1, N).repeat(N, 1).reshape(1, N, N, 1)
    # Negative Laplacian in Fourier space
    lap = (k_x ** 2 + k_y ** 2)
    lap[0, 0, 0, 0] = 1.0
    f_h = w_h / lap

    ux_h = 2 * np.pi / L * 1j * k_y * f_h
    uy_h = -2 * np.pi / L * 1j * k_x * f_h

    ux = torch.fft.irfft2(ux_h[:, :, :k_max + 1], dim=[1, 2])
    uy = torch.fft.irfft2(uy_h[:, :, :k_max + 1], dim=[1, 2])
    return ux, uy

if __name__ == '__main__':
    try: data = np.load(r"C:\Users\Noahc\Documents\USYD\PHD\8 - Github\PINO datasets\NS_fft_Re500_T4000.npy" ) 
    except: 
        try: data = np.load(r'/project/MLFluids/NS_fft_Re500_T4000.npy')
        except: raise('No Dataset File Found')
    else: print('Dataset Allocated')
    
    print('Input Shape: ', data.shape)
    data = torch.tensor(data).permute(0,2,3,1)
    sol_cartesian = np.zeros((data.shape[0], data.shape[1], data.shape[2], data.shape[3], 2))
    sol_cartesian[: , : , : , :, 0], sol_cartesian[: , : , : , :, 1] = vor2vel(data)
    sol_cartesian = torch.tensor(sol_cartesian).permute(0,3,1,2,4)
    print('Output Shape: ', sol_cartesian.shape)
    sol_cartesian = sol_cartesian.numpy()
    np.save('NS_fft_Re500_T4000_cartesian.npy', sol_cartesian)
    print('Conversion Complete')

    # data = np.load(r'/project/MLFluids/NS_fft_Re500_T4000_cartesian.npy')
    # print('Dataset Allocated')

    # data1 = torch.tensor(data, dtype=torch.float)
    # print(data1.shape)