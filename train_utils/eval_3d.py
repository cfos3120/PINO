import torch
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm
from timeit import default_timer

from .losses import LpLoss, PINO_loss3d

def eval_ns(model,  # model
            loader,  # dataset instance
            dataloader,  # dataloader
            forcing,  # forcing
            config,  # configuration dict
            device,  # device id
            log=False,
            project='PINO-default',
            group='FDM',
            tags=['Nan'],
            use_tqdm=True):
    '''
    Evaluate the model for Navier Stokes equation
    '''

    # data parameters
    v = 1 / config['data']['Re']
    S, T = loader.S, loader.T
    t_interval = config['data']['time_interval']
    # eval settings
    batch_size = config['test']['batchsize']
    
    # add extra lp_loss_factor for sensitivity test
    lp_loss_factor = config['train']['pde_loss_factor']
    print('Loss Factor Used: ', lp_loss_factor)

    # intitialise training lsit savers
    loss_ic_list = list()
    loss_l2_list = list()
    loss_f_list = list()

    model.eval()
    myloss = LpLoss(size_average=True)
    if use_tqdm:
        pbar = tqdm(dataloader, dynamic_ncols=True, smoothing=0.05)
    else:
        pbar = dataloader
    loss_dict = {'train_f': 0.0,
                 'test_l2': 0.0,
                 'loss_ic': 0.0}
    start_time = default_timer()
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            x_in = F.pad(x, (0, 0, 0, 5), "constant", 0)
            out = model(x_in).reshape(batch_size, S, S, T + 5, 2)
            out = out[..., :-5, :]
            #x = x[:, :, :, 0, -1]
            
            loss_l2 = myloss(out.view(batch_size, S, S, T, 2),
                             y.view(batch_size, S, S, T, 2))
            #loss_l2 = myloss(out.view(batch_size, S, S, T), y.view(batch_size, S, S, T))
            
            loss_ic, loss_f = PINO_loss3d(out.view(batch_size, S, S, T, 2),
                                              x, forcing,
                                              v, t_interval, lp_loss_factor)
            #loss_ic, loss_f = PINO_loss3d(out.view(batch_size, S, S, T), x, forcing, v, t_interval)

            loss_dict['train_f'] += loss_f
            loss_dict['test_l2'] += loss_l2
            loss_dict['loss_ic'] += loss_ic

                    # print losses to file
            loss_ic_list.append(loss_ic)
            loss_l2_list.append(test_l2)
            loss_f_list.append(loss_f)

    end_time = default_timer()
    test_l2 = loss_dict['test_l2'].item() / len(dataloader)
    loss_f = loss_dict['train_f'].item() / len(dataloader)
    loss_ic = loss_dict['loss_ic'].item() / len(dataloader)
    print(f'==Averaged relative L2 error is: {test_l2}==\n'
          f'==Averaged equation error is: {loss_f}==\n'
          f'==Averaged intial condition error is: {loss_ic}==')
    print(f'Time cost: {end_time - start_time} s')
    print(f'Number of Cases: {len(dataloader)} s')

    losses_total = [loss_ic_list,loss_l2_list,loss_f_list]

    np.save(config['test']['ckpt'][:-3] + '_val_losses', losses_total)
    print('Losses saved at %s' % config['test']['ckpt'][-3] + '_losses \n \n')