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

# parse options
parser = ArgumentParser(description='Basic paser')
parser.add_argument('--config_path', type=str, help='Path to the configuration file')
parser.add_argument('--log', action='store_true', help='Turn on the wandb')
args = parser.parse_args()

print('Finished Parsing')

config_file = args.config_path
with open(config_file, 'r') as stream:
    config = yaml.load(stream, yaml.FullLoader)

print('Finished Configuration')