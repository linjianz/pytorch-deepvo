#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Linjian Zhang
Email: linjian93@foxmail.com
Creat Time: 2017-12-27 16:20:53
Program: 
Description:
"""
import torch
from torch.utils.data import DataLoader
from time import time
from tqdm import tqdm
from net.cnn import *
from dataset.kitti import KITTIDataSet
from utils.post_process import cal_absolute_from_relative, plot_from_pose
from utils.misc import to_var, adjust_learning_rate, pre_create_file_train, pre_create_file_test
import numpy as np
import math
import argparse
from tensorboardX import SummaryWriter
import re
import os

parser = argparse.ArgumentParser()
parser.add_argument('--server', default=None, type=int, help='server')
parser.add_argument('--phase', default=None, help='Train or Test')
parser.add_argument('--resume', default=None, help='Resume or scratch')

# resume training or test
parser.add_argument('--net_restore', default=None, help='Restore net name')
parser.add_argument('--dir_restore', default=None, help='Restore file name')
parser.add_argument('--model_restore', default=None, help='Restore model-id')

parser.add_argument('--net_name', default=None, help='different net with different name')
parser.add_argument('--dir0', default=None, help='change it every time when you run the code')
parser.add_argument('--batch_size', default=32, type=int, help='batch size')
parser.add_argument('--epoch_max', default=200, type=int, help='max epoch numbers')
parser.add_argument('--epoch_test', default=5, type=int, help='test a complete sequence')
parser.add_argument('--epoch_save', default=5, type=int, help='max epoch numbers')
parser.add_argument('--lr_base', default=1e-4, type=float, help='base learning rate')
parser.add_argument('--lr_decay_rate', default=0.5, type=float, help='decay rate')
parser.add_argument('--epoch_lr_decay', default=40, type=int, help='every # epoch, lr decay 0.5')
parser.add_argument('--beta', default=50, type=int, help='loss = loss_t + beta * loss_r')

parser.add_argument("--gpu", default='0', help='GPU id list')
parser.add_argument("--workers", default=4, type=int, help='workers numbers')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
gpu_list = re.split('[, ]', args.gpu)
args.gpu = range(len(list(filter(None, gpu_list))))
args.workers = int(args.workers)
args.batch_size = int(args.batch_size)


if args.server == 6099:
    dir_data = '/media/csc105/Data/dataset-jiange/data_odometry_color/sequences'
    dir_label = 'dataset'
elif args.server == 6199:
    dir_data = '/media/Data/dataset_jiange/data_odometry_color/sequences'
    dir_label = 'dataset'
elif args.server == 6499:
    dir_data = '/media/jiange/095df4a3-d72c-43d9-bfbd-e78651afba19/dataset-jiange/data_odometry_color/sequences'
    dir_label = 'dataset'
else:
    raise Exception('Must give the right server id!')


def run_batch(sample, model, loss_func, optimizer=None, phase=None):
    if phase == 'Train':
        model.train()
    else:
        model.eval()

    img = to_var(sample['img'])         # [bs, 6, H, W]
    label_pre = model(img)

    if phase == 'Train' or phase == 'Valid':
        label = to_var(sample['label'])  # [bs, 6]
        loss1 = loss_func(label_pre[:, :3], label[:, :3])
        loss2 = loss_func(label_pre[:, 3:], label[:, 3:])
        loss = loss1 + args.beta * loss2

        loss_x = loss_func(label_pre[:, 0], label[:, 0])
        loss_y = loss_func(label_pre[:, 1], label[:, 1])
        loss_z = loss_func(label_pre[:, 2], label[:, 2])
        loss_tx = loss_func(label_pre[:, 3], label[:, 3])
        loss_ty = loss_func(label_pre[:, 4], label[:, 4])
        loss_tz = loss_func(label_pre[:, 5], label[:, 5])

        if phase == 'Train':
            optimizer.zero_grad()           # clear gradients for this training step
            loss.backward()                 # bp, compute gradients
            optimizer.step()                # apply gradients

        return loss.data[0], loss1.data[0], loss2.data[0], label_pre.data, \
            loss_x.data[0], loss_y.data[0], loss_z.data[0], loss_tx.data[0], loss_ty.data[0], loss_tz.data[0]
    else:
        return label_pre.data


def run_val(model, loss_func, loader):
    """
    evaluate multi-batches
    :param model: model
    :param loss_func: MSELoss
    :param loader: loader_v
    :return: mean loss
    """
    loss_ret = []
    loss1_ret = []
    loss2_ret = []
    loss_x_ret = []
    loss_y_ret = []
    loss_z_ret = []
    loss_tx_ret = []
    loss_ty_ret = []
    loss_tz_ret = []
    for _, sample_v in enumerate(loader):
        loss_v, loss1_v, loss2_v, _, loss_x_v, loss_y_v, loss_z_v, loss_tx_v, loss_ty_v, loss_tz_v = \
            run_batch(sample=sample_v, model=model, loss_func=loss_func, phase='Valid')
        loss_ret.append(loss_v)
        loss1_ret.append(loss1_v)
        loss2_ret.append(loss2_v)
        loss_x_ret.append(loss_x_v)
        loss_y_ret.append(loss_y_v)
        loss_z_ret.append(loss_z_v)
        loss_tx_ret.append(loss_tx_v)
        loss_ty_ret.append(loss_ty_v)
        loss_tz_ret.append(loss_tz_v)
    loss_mean = np.mean(loss_ret)
    loss1_mean = np.mean(loss1_ret)
    loss2_mean = np.mean(loss2_ret)
    loss_x_mean = np.mean(loss_x_ret)
    loss_y_mean = np.mean(loss_y_ret)
    loss_z_mean = np.mean(loss_z_ret)
    loss_tx_mean = np.mean(loss_tx_ret)
    loss_ty_mean = np.mean(loss_ty_ret)
    loss_tz_mean = np.mean(loss_tz_ret)

    return loss_mean, loss1_mean, loss2_mean, loss_x_mean, loss_y_mean, loss_z_mean, \
        loss_tx_mean, loss_ty_mean, loss_tz_mean


def run_test(model, loss_func, seq, dir_model=None, epoch=None, dir_time=None, is_testing=True):
    print('\nTest sequence {:02d} >>>'.format(seq))
    data_set = KITTIDataSet(dir_data=dir_data, dir_label=dir_label, phase='Test', seq=seq)
    loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    pose_ret = []
    for _, sample_batch in enumerate(tqdm(loader)):
        pose_pre = run_batch(sample=sample_batch, model=model, loss_func=loss_func, phase='Test')
        pose_ret.extend(pose_pre.cpu().numpy())

    if is_testing:
        print('Save pose in {:s}'.format(dir_time))
        np.savetxt(dir_time+'/pose_{:d}.txt'.format(seq), pose_ret)
        cal_absolute_from_relative(seq, dir_test=dir_time, is_testing=is_testing)
    else:
        print('Calculate absolute pose')
        pose_abs = cal_absolute_from_relative(seq, xyz_euler=pose_ret, is_testing=is_testing)
        print('Plot trajectory')
        plot_from_pose(seq=seq, dir_save=dir_model, pose_abs=pose_abs, epoch=epoch)

    del data_set


def main():
    torch.set_default_tensor_type('torch.FloatTensor')
    model = CNN()
    if torch.cuda.is_available():
        model = nn.DataParallel(model.cuda(), device_ids=args.gpu)

    if (args.phase == 'Train' and args.resume == 'Yes') or args.phase == 'Test':
        dir_restore = 'model/' + args.net_restore + '/' + args.dir_restore + '/' + args.model_restore + '.pkl'
        print('\nRestore from {:s}'.format(dir_restore))
        model.load_state_dict(torch.load(dir_restore))

    if args.phase == 'Train':
        if args.resume == 'No':
            print('\nInitialize from scratch')
        dir_model, dir_log = pre_create_file_train(args)
        data_set_t = KITTIDataSet(dir_data=dir_data, dir_label=dir_label, phase='Train')
        loader_t = DataLoader(data_set_t, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
        # data_set_v = KITTIDataSet(dir_data=dir_data, dir_label=dir_label, phase='Val')
        # loader_v = DataLoader(data_set_v, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

        loss_func = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_base)
        step_per_epoch = int(math.floor(len(data_set_t)/loader_t.batch_size))
        # step_val = int(math.floor(step_per_epoch / 3))

        writer = SummaryWriter(dir_log)
        for epoch in np.arange(args.epoch_max):
            adjust_learning_rate(optimizer, epoch, args.lr_base,
                                 gamma=args.lr_decay_rate,
                                 epoch_lr_decay=args.epoch_lr_decay)

            # plot trajectory
            # if epoch % args.epoch_test == 0:
            #     run_test(model, loss_func, seq=9, dir_model=dir_model, epoch=epoch, is_testing=False)
            #     run_test(model, loss_func, seq=5, dir_model=dir_model, epoch=epoch, is_testing=False)

            for step, sample_t in enumerate(loader_t):
                step_global = epoch * step_per_epoch + step
                tic = time()
                loss, loss1, loss2, _, loss_x, loss_y, loss_z, loss_tx, loss_ty, loss_tz = \
                    run_batch(sample=sample_t, model=model, loss_func=loss_func, optimizer=optimizer, phase='Train')
                hour_per_epoch = step_per_epoch * ((time() - tic) / 3600)

                # display and add to tensor board
                if (step+1) % 10 == 0:
                    print('\n{:.3f} [{:03d}/{:03d}] [{:03d}/{:03d}] lr {:.6f} L {:.4f}={:.4f}+{:d}*{:.4f} '
                          '[{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}]'.
                          format(hour_per_epoch, epoch+1, args.epoch_max, step+1, step_per_epoch,
                                 optimizer.param_groups[0]['lr'], loss, loss1, args.beta, loss2,
                                 loss_x, loss_y, loss_z, loss_tx, loss_ty, loss_tz))
                    writer.add_scalars('./train',
                                       {'loss_t': loss, 'loss1_t': loss1, 'loss2_t': loss2, 'loss_x_t': loss_x,
                                        'loss_y_t': loss_y, 'loss_z_t': loss_z, 'loss_tx_t': loss_tx,
                                        'loss_ty_t': loss_ty, 'loss_tz_t': loss_tz},
                                       step_global)

                # if (step+1) % step_val == 0:
                #     batch_v = int(math.ceil(len(data_set_v)/loader_v.batch_size))
                #     loss_v, loss1_v, loss2_v, loss_x_v, loss_y_v, loss_z_v, loss_tx_v, loss_ty_v, loss_tz_v = \
                #         run_val(model, loss_func, loader_v)
                #     print('\n{:d} batches: L {:.4f}={:.4f}+{:d}*{:.4f} [{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}]'.
                #          format(batch_v, loss_v, loss1_v, args.beta, loss2_v, loss_x_v, loss_y_v, loss_z_v, loss_tx_v,
                #                  loss_ty_v, loss_tz_v))
                #     writer.add_scalars('./train-val',
                #                       {'loss_v': loss_v, 'loss1_v': loss1_v, 'loss2_v': loss2_v, 'loss_x_v': loss_x_v,
                #                         'loss_y_v': loss_y_v, 'loss_z_v': loss_z_v, 'loss_tx_v': loss_tx_v,
                #                         'loss_ty_v': loss_ty_v, 'loss_tz_v': loss_tz_v},
                #                        step_global)

            # save
            if (epoch+1) % args.epoch_save == 0:
                print('\nSaving model: {:s}/model-{:d}.pkl'.format(dir_model, epoch+1))
                torch.save(model.state_dict(), (dir_model + '/model-{:d}.pkl'.format(epoch+1)))

    if args.phase == 'Test':
        dir_time = pre_create_file_test(args)
        loss_func = nn.MSELoss()
        for seq in range(11):
            run_test(model, loss_func, seq=seq, dir_time=dir_time, is_testing=True)


if __name__ == '__main__':
    main()
