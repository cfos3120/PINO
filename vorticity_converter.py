# Script to convert Vorticity to Cartesian Velocity Components
# Author: Noah Foster

import yaml
from argparse import ArgumentParser
import math
import torch
import numpy as np
print('Finished Importing Libaries')

# Declare conversion function (sourced from original PINO library)
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
    
    # This script is used for converting existing datasets from vorticity from to Cartesian Velocities
    
    parser = ArgumentParser(description='Basic paser')
    parser.add_argument('--dataset', type=str, help='Dataset type (e.g. "train", "test1", "test2" or "tune")')
    args = parser.parse_args()
    
    # Envrionment Selection process:
    # 1. try to source local computer dataset
    local_path = "C:/Users/Noahc/Documents/USYD/PHD/8 - Github/PINO datasets/"
    # 2. try to source dataset located on Artemis 
    artemis_path = "/project/MLFluids/"
    
    # Dataste type allocation 
    if args.dataset == "train":
        dataset_name = "NS_fft_Re500_T4000.npy"
    elif args.dataset == "test1":
        dataset_name = "NS_Re500_s256_T100_test.npy"
    elif args.dataset == "test2":
        dataset_name = "NS_fine_Re500_T64_R256_sub1.npy"
    elif args.dataset == "tune":
        dataset_name = " "
    else: raise Exception('Sorry, dataset type not eligible (e.g. "train", "test" or "tune")')
        
    # Load in Datatype (try local environment first)
    try: data = np.load(local_path + dataset_name)[:2, ...] 
    except: 
        try: data = np.load(artemis_path + dataset_name)
        except: raise Exception('No Dataset File Found')
    else: print('Dataset Found and Allocated')
    
    # Conversion process
    print('Input Shape: ', data.shape)
    data = torch.tensor(data).permute(0,2,3,1)
    sol_cartesian = np.zeros((data.shape[0], data.shape[1], data.shape[2], data.shape[3], 2))
    sol_cartesian[: , : , : , :, 0], sol_cartesian[: , : , : , :, 1] = vor2vel(data)
    sol_cartesian = torch.tensor(sol_cartesian).permute(0,3,1,2,4)
    print('Output Shape: ', sol_cartesian.shape)
    sol_cartesian = sol_cartesian.numpy()
    
    # Save and Validate Numpy File
    try:
        np.save(artemis_path+dataset_name[:-4]+'_cartesian', sol_cartesian)
        print('Conversion Complete')
    except: print('Numpy Save Failed OR not saved due to local environment')

    try:
        np.load(artemis_path+dataset_name[:-4]+'_cartesian.npy')
        print('Dataset allocated and Numpy Loading Works')
    except: print('Numpy Load Failed OR not saved due to local environment')