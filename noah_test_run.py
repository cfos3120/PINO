print('Hello World')

import yaml
from argparse import ArgumentParser
import math
import torch
from torch.utils.data import DataLoader

from solver.random_fields import GaussianRF
from train_utils import Adam
from train_utils.datasets import NSLoader, online_loader, DarcyFlow
from train_utils.train_3d import mixed_train
from train_utils.train_2d import train_2d_operator
from models import FNN3d, FNN2d

print('Finished Importing Libaries')