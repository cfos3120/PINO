print('Hello World')

import yaml
from argparse import ArgumentParser
import math
import torch
import numpy as np
from train_utils.utils import vor2vel
#from torch.utils.data import DataLoader

#from solver.random_fields import GaussianRF
#from train_utils import Adam
#from train_utils.datasets import NSLoader, online_loader, DarcyFlow
#from train_utils.train_3d import mixed_train
#from train_utils.train_2d import train_2d_operator
#from models import FNN3d, FNN2d

print('Finished Importing Libaries')

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
    print('Conversion Complete')