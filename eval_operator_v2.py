# Model Evaluation Script
# Author: Noah Foster

# Import Libraries
import yaml

import numpy as np
import torch
from torch.utils.data import Eval_loader
import torch.nn.functional as F
print('Pytorch version: ',torch.__version__)

from tqdm import tqdm
from timeit import default_timer

from train_utils.losses import LpLoss, PINO_loss3d
from models import FNN3d, FNN2d
from train_utils import NSLoader

from argparse import ArgumentParser

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 1. Parse arguments and load configurations
    parser = ArgumentParser(description='Basic paser')
    parser.add_argument('--config_path', type=str, help='Path to the configuration file')
    options = parser.parse_args()
    config_file = options.config_path
    with open(config_file, 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)
    
    # 2. Load in Dataset and create `Loader`
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
    
    print(f'Resolution : {loader.S}x{loader.S}x{loader.T}')

    # 3. Create model and load in model version
    model = FNN3d(modes1=config['model']['modes1'],
                  modes2=config['model']['modes2'],
                  modes3=config['model']['modes3'],
                  fc_dim=config['model']['fc_dim'],
                  layers=config['model']['layers'],
                  in_dim=5,
                  out_dim=2).to(device)

    if 'ckpt' in config['test']:
        ckpt_path = config['test']['ckpt']
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['model'])
        print('Weights loaded from %s' % ckpt_path)
    
    # 4. Data and evaluation parameters
    v = 1 / config['data']['Re']
    S, T = loader.S, loader.T
    t_interval = config['data']['time_interval']
    batch_size = config['test']['batchsize']

    # 5. Get forcing function and loss function
    pi_meshgrid = torch.tensor(np.linspace(0, 2*np.pi, S+1)[:-1], dtype=torch.float).reshape(1, S).repeat(S, 1)
    forcing = -4 * torch.cos(4*(pi_meshgrid)).reshape(1,S,S,1).to(device)

    myloss = LpLoss(size_average=True)

    # 6. Intitialise training list disctionaries
    loss_ic_list = list()
    loss_l2_list = list()
    loss_f_list = list()
    loss_c_list = list()
    loss_m1_list = list()
    loss_m2_list = list()

    # 7. Evaluate Model for test cases
    model.eval()
    start_time = default_timer()
    with torch.no_grad():
        for x, y in eval_loader:
            x, y = x.to(device), y.to(device)
            x_in = F.pad(x, (0, 0, 0, 5), "constant", 0)
            out = model(x_in).reshape(batch_size, S, S, T + 5, 2)
            out = out[..., :-5, :]
            
            loss_l2 = myloss(out.view(batch_size, S, S, T, 2),
                             y.view(batch_size, S, S, T, 2))
            
            loss_ic, loss_f, loss1, loss2, loss3 = PINO_loss3d(out.view(batch_size, S, S, T, 2),
                                                                x, forcing,
                                                                v, t_interval)

            loss_ic_list.append(loss_ic.item())
            loss_l2_list.append(loss_l2.item())
            loss_f_list.append(loss_f.item())
            loss_c_list.append(loss1.item())
            loss_m1_list.append(loss2.item())
            loss_m2_list.append(loss3.item())
    
    end_time = default_timer()

    # 8. Finalize Loss disctionaries and print outputs
    loss_l2 = sum(loss_l2_list) / len(eval_loader)
    loss_f = sum(loss_f_list) / len(eval_loader)
    loss_ic = sum(loss_ic_list) / len(eval_loader)
    losses_total = [loss_ic_list,loss_l2_list,loss_f_list, loss_c_list, loss_m1_list, loss_m2_list]
    
    print(f'==Averaged relative L2 error is: {loss_l2}==\n'
          f'==Averaged equation error is: {loss_f}==\n'
          f'==Averaged intial condition error is: {loss_ic}==')
    print(f'Time cost: {end_time - start_time} s')
    print(f'Number of Cases: {len(eval_loader)} s')

    print('Losses saving at %s' % config['test']['ckpt'][-3] + '_losses')
    np.save(config['test']['ckpt'][:-3] + '_val_losses', losses_total)
    print('Evaluation and Save Complete \n \n')