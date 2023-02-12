import yaml

import torch
from torch.utils.data import DataLoader
from models import FNN3d, FNN2d
from train_utils import NSLoader, get_forcing##, DarcyFlow

from train_utils.eval_3d import eval_ns
##from train_utils.eval_2d import eval_darcy

from argparse import ArgumentParser



def test_3d(config):
    device = 0 if torch.cuda.is_available() else 'cpu'
    data_config = config['data']
    loader = NSLoader(datapath1=data_config['datapath'],
                      nx=data_config['nx'], nt=data_config['nt'],
                      sub=data_config['sub'], sub_t=data_config['sub_t'],
                      N=data_config['total_num'],
                      t_interval=data_config['time_interval'])

    eval_loader = loader.make_loader(n_sample=data_config['n_sample'],
                                     batch_size=config['test']['batchsize'],
                                     start=data_config['offset'],
                                     train=data_config['shuffle'])
    
    # create model
    model = FNN3d(modes1=config['model']['modes1'],
                  modes2=config['model']['modes2'],
                  modes3=config['model']['modes3'],
                  fc_dim=config['model']['fc_dim'],
                  layers=config['model']['layers'],
                  in_dim=5,
                  out_dim=2).to(device)

    # Load from checkpoint
    if 'ckpt' in config['test']:
        ckpt_path = config['test']['ckpt']
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['model'])
        print('Weights loaded from %s' % ckpt_path)
    print(f'Resolution : {loader.S}x{loader.S}x{loader.T}')
    
    forcing = get_forcing(loader.S).to(device)
    eval_ns(model,
            loader,
            eval_loader,
            forcing,
            config,
            device=device)

if __name__ == '__main__':
    parser = ArgumentParser(description='Basic paser')
    parser.add_argument('--config_path', type=str, help='Path to the configuration file')
    parser.add_argument('--log', action='store_true', help='Turn on the wandb')
    options = parser.parse_args()
    config_file = options.config_path
    with open(config_file, 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)

    test_3d(config)
