#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Linjian Zhang
Email: linjian93@foxmail.com
Create Time: 2017-12-17 17:26:34
Program: 
Description: 
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import numpy as np
import os

# experiment1-cnn
loss_t = '/home/jiange/Downloads/run-20171214_10-train-val-loss_t-tag-.-train-val.json'
loss_v = '/home/jiange/Downloads/run-20171214_10-train-val-loss_v-tag-.-train-val.json'
loss1_t = '/home/jiange/Downloads/run-20171214_10-train-val-loss1_t-tag-.-train-val.json'
loss1_v = '/home/jiange/Downloads/run-20171214_10-train-val-loss1_v-tag-.-train-val.json'
loss2_t = '/home/jiange/Downloads/run-20171214_10-train-val-loss2_t-tag-.-train-val.json'
loss2_v = '/home/jiange/Downloads/run-20171214_10-train-val-loss2_v-tag-.-train-val.json'
loss_tx_t = '/home/jiange/Downloads/run-20171214_10-train-val-loss_tx_t-tag-.-train-val.json'
loss_tx_v = '/home/jiange/Downloads/run-20171214_10-train-val-loss_tx_v-tag-.-train-val.json'
loss_ty_t = '/home/jiange/Downloads/run-20171214_10-train-val-loss_ty_t-tag-.-train-val.json'
loss_ty_v = '/home/jiange/Downloads/run-20171214_10-train-val-loss_ty_v-tag-.-train-val.json'
loss_tz_t = '/home/jiange/Downloads/run-20171214_10-train-val-loss_tz_t-tag-.-train-val.json'
loss_tz_v = '/home/jiange/Downloads/run-20171214_10-train-val-loss_tz_v-tag-.-train-val.json'
loss_x_t = '/home/jiange/Downloads/run-20171214_10-train-val-loss_x_t-tag-.-train-val.json'
loss_x_v = '/home/jiange/Downloads/run-20171214_10-train-val-loss_x_v-tag-.-train-val.json'
loss_y_t = '/home/jiange/Downloads/run-20171214_10-train-val-loss_y_t-tag-.-train-val.json'
loss_y_v = '/home/jiange/Downloads/run-20171214_10-train-val-loss_y_v-tag-.-train-val.json'
loss_z_t = '/home/jiange/Downloads/run-20171214_10-train-val-loss_z_t-tag-.-train-val.json'
loss_z_v = '/home/jiange/Downloads/run-20171214_10-train-val-loss_z_v-tag-.-train-val.json'

# dir_t = [loss_t, loss1_t, loss2_t, loss_x_t, loss_y_t, loss_z_t, loss_tx_t, loss_ty_t, loss_tz_t]
# dir_v = [loss_v, loss1_v, loss2_v, loss_x_v, loss_y_v, loss_z_v, loss_tx_v, loss_ty_v, loss_tz_v]
# name_title = ['loss', 'translation', 'rotation', 'x', 'y', 'z', '$\psi$', '$\chi$', '$\phi$']
# dir_save = 'loss/cnn-vo-cons'

# experiment1-cnn-lstm
loss_t_1 = 'loss-json/run_20171224_1_train-tag-loss.json'
loss_v_1 = 'loss-json/run_20171224_1_val-tag-loss.json'
loss1_t_1 = 'loss-json/run_20171224_1_train-tag-loss1.json'
loss1_v_1 = 'loss-json/run_20171224_1_val-tag-loss1.json'
loss2_t_1 = 'loss-json/run_20171224_1_train-tag-loss2.json'
loss2_v_1 = 'loss-json/run_20171224_1_val-tag-loss2.json'
# dir_t = [loss_t_1, loss1_t_1, loss2_t_1]
# dir_v = [loss_v_1, loss1_v_1, loss2_v_1]
# name_title = ['loss', 'translation', 'rotation']
# dir_save = 'loss/experiment1-cnn-lstm'

# experiment1-cnn-lstm-cons
loss_t_2 = 'loss-json/cnn-lstm-vo-cons/run_train-val_loss_t-tag-._train-val.json'
loss_v_2 = 'loss-json/cnn-lstm-vo-cons/run_train-val_loss_v-tag-._train-val.json'
loss1_t_2 = 'loss-json/cnn-lstm-vo-cons/run_train-val_loss1_t-tag-._train-val.json'
loss1_v_2 = 'loss-json/cnn-lstm-vo-cons/run_train-val_loss1_v-tag-._train-val.json'
loss2_t_2 = 'loss-json/cnn-lstm-vo-cons/run_train-val_loss2_t-tag-._train-val.json'
loss2_v_2 = 'loss-json/cnn-lstm-vo-cons/run_train-val_loss2_v-tag-._train-val.json'
# dir_t = [loss_t_2, loss1_t_2, loss2_t_2]
# dir_v = [loss_v_2, loss1_v_2, loss2_v_2]
# name_title = ['loss', 'translation', 'rotation']
# dir_save = 'loss/experiment1-cnn-lstm-cons'

# experiment2-cnn-lstm
loss1_t_3 = 'loss-json/experiment2-cnn-lstm/run-train-loss1_t-tag-.-train.json'
loss2_t_3 = 'loss-json/experiment2-cnn-lstm/run-train-loss2_t-tag-.-train.json'
dir_t = [loss1_t_3]
dir_v = [loss2_t_3]
name_title = ['loss']
dir_save = 'loss/experiment2-cnn-lstm'


if not os.path.exists(dir_save):
    os.makedirs(dir_save)


def plot_train_val():
    for dir1, dir2, name in zip(dir_t, dir_v, name_title):
        plt.close('all')
        data_t = np.array(json.load(open(dir1, 'r')))
        data_v = np.array(json.load(open(dir2, 'r')))
        x_t = data_t[5::2, 1]
        y_t = data_t[5::2, 2]
        x_v = data_v[5::2, 1]
        y_v = data_v[5::2, 2]
        plt.plot(x_t, y_t, '-b', label='Train loss1')
        plt.plot(x_v, y_v, '-r', label='Train loss2')

        if name != 'loss':
            plt.title('Loss of {:s}'.format(name))
        else:
            plt.title('Train loss on sequence 00-10')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid('on')
        plt.savefig(dir_save + '/{:s}.png'.format(name))
        # plt.show()


if __name__ == '__main__':
    plot_train_val()
