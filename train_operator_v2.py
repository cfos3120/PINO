# Model Training Script
# Author: Noah Foster

# Import Libraries
import yaml

import numpy as np
import torch
import torch.nn.functional as F
print('Pytorch version: ',torch.__version__)

from train_utils import Adam
from train_utils.losses import LpLoss, PINO_loss3d
from train_utils.utils import save_checkpoint
from train_utils.data_utils import sample_data
from train_utils.datasets import NSLoader
from models import FNN3d

from timeit import default_timer

from argparse import ArgumentParser

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 1. Parse arguments and load configurations
    parser = ArgumentParser(description='Basic paser')
    parser.add_argument('--config_path', type=str, help='Path to the configuration file')
    args = parser.parse_args()
    config_file = args.config_path
    with open(config_file, 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)
    print(config)

    # 2. Load in Dataset and create `Loader`
    data_config = config['data']
    loader = NSLoader(datapath1=data_config['datapath'],
                          nx=data_config['nx'], nt=data_config['nt'],
                          sub=data_config['sub'], sub_t=data_config['sub_t'],
                          N=data_config['total_num'],
                          t_interval=data_config['time_interval'])

    train_loader = loader.make_loader(data_config['n_sample'],
                                      batch_size=config['train']['batchsize'],
                                      start=data_config['offset'],
                                      train=data_config['shuffle'])

    train_loader = sample_data(train_loader)

    print(f'Resolution : {loader.S}x{loader.S}x{loader.T}')

    # 3. Create model and load in model version
    model = FNN3d(modes1=config['model']['modes1'],
                  modes2=config['model']['modes2'],
                  modes3=config['model']['modes3'],
                  fc_dim=config['model']['fc_dim'],
                  layers=config['model']['layers'],
                  in_dim=5,
                  out_dim=2).to(device)

    if 'ckpt' in config['train']:
        ckpt_path = config['train']['ckpt']
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['model'])
        print('Weights loaded from %s' % ckpt_path)

    # 4. Training optimizer and learning rate scheduler
    optimizer = Adam(model.parameters(), betas=(0.9, 0.999),
                     lr=config['train']['base_lr'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=config['train']['milestones'],
                                                     gamma=config['train']['scheduler_gamma'])
    
    # 5. Data and training parameters
    v = 1 / config['data']['Re']
    S, T = loader.S, loader.T
    t_interval = config['data']['time_interval']
    batch_size = config['test']['batchsize']
    batch_size = config['train']['batchsize']
    ic_weight = config['train']['ic_loss']
    f_weight = config['train']['f_loss']
    xy_weight = config['train']['xy_loss']
    num_data_iter = config['train']['data_iter']
    num_eqn_iter = config['train']['eqn_iter']
    epochs = config['train']['epochs']

    # 6. Get forcing function and loss function
    pi_meshgrid = torch.tensor(np.linspace(0, 2*np.pi, S+1)[:-1], dtype=torch.float).reshape(1, S).repeat(S, 1)
    forcing = -4 * torch.cos(4*(pi_meshgrid)).reshape(1,S,S,1).to(device)
    myloss = LpLoss(size_average=True)

    # 7. Intitialise training list disctionaries

    loss_ic_list = list()
    loss_l2_list = list()
    loss_f_list = list()
    loss_c_list = list()
    loss_m1_list = list()
    loss_m2_list = list()
    loss_total_list = list()
    epoch_timer_list = list()

    # 8. Train Model with training cases
    for epoch in range(epochs):
        model.train()        
        epoch_start_time = default_timer()

        loss_ic_sum = 0.0
        loss_l2_sum = 0.0
        loss_f_sum = 0.0
        loss_c_sum = 0.0
        loss_m1_sum = 0.0
        loss_m2_sum = 0.0
        loss_total_sum = 0.0

        for _ in range(num_data_iter):
            x, y = next(train_loader)
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            x_in = F.pad(x, (0, 0, 0, 5), "constant", 0)
            out = model(x_in).reshape(batch_size, S, S, T + 5, 2)
            out = out[:, :, :, :-5, :]

            loss_l2 = myloss(out.view(batch_size, S, S, T, 2),
                             y.view(batch_size, S, S, T, 2))

            if ic_weight != 0 or f_weight != 0:
                loss_ic, loss_f, loss1, loss2, loss3 = PINO_loss3d(out.view(batch_size, S, S, T, 2),
                                                                x, forcing,
                                                                v, t_interval)
            else:
                loss_ic, loss_f = torch.zeros(1).to(device), torch.zeros(1).to(device)

            total_loss = loss_l2 * xy_weight + loss_f * f_weight + loss_ic * ic_weight

            total_loss.backward()
            optimizer.step()

            loss_ic_sum += loss_ic.item()
            loss_l2_sum += loss_l2.item()
            loss_f_sum += loss_f.item()
            loss_c_sum += loss1.item()
            loss_m1_sum += loss2.item()
            loss_m2_sum += loss3.item()
            loss_total_sum += total_loss.item()

        if num_data_iter != 0:
            loss_ic_sum /= num_data_iter
            loss_l2_sum /= num_data_iter
            loss_f_sum /= num_data_iter
            loss_c_sum /= num_data_iter
            loss_m1_sum /= num_data_iter
            loss_m2_sum /= num_data_iter
            loss_total_sum /= num_data_iter

        for _ in range(num_eqn_iter):
            continue

        scheduler.step()
        epoch_end_time = default_timer()

        loss_ic_list.append(loss_ic_sum)
        loss_l2_list.append(loss_l2_sum)
        loss_f_list.append(loss_f_sum)
        loss_c_list.append(loss_c_sum)
        loss_m1_list.append(loss_m1_sum)
        loss_m2_list.append(loss_m2_sum)
        loss_total_list.append(loss_total_sum)
        epoch_timer_list.append(epoch_end_time-epoch_start_time)

    # 9. Finalize Loss disctionaries and print outputs
    losses_total = [loss_ic_list,
                    loss_l2_list,
                    loss_f_list,
                    loss_c_list,
                    loss_m1_list,
                    loss_m2_list,
                    loss_total_list,
                    epoch_timer_list]
    
    print(f'==Final Averaged intial condition error is: {loss_ic_sum}==\n'
          f'==Final Averaged relative L2 error is: {loss_l2_sum}==\n'
          f'==Final Averaged equation error is: {loss_f_sum}==\n'
          f'==Final Averaged continuity eqn error is: {loss_c_sum}==\n'
          f'==Final Averaged x-momentum eqn error is: {loss_m1_sum}==\n'
          f'==Final Averaged y-momentum eqn error is: {loss_m2_sum}==\n'
          f'==Final Averaged total weighted error is: {loss_total_sum}==\n'
          f'==Final Total Training Time was: {sum(epoch_timer_list)}==\n')
    
    print('Saving trained Model')
    save_checkpoint(config['train']['save_dir'],
                    config['train']['save_name'],
                    model, optimizer)

    print('Losses saving at %s' % config['train']['ckpt'][-3] + '_losses')
    np.save(config['train']['ckpt'][:-3] + '_training_losses', losses_total)
    print('Training and Save Complete \n \n')