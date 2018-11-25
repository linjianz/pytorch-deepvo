#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Linjian Zhang
Email: linjian93@foxmail.com
Create Time: 2017-11-21 15:42:03
Program: 
Description: 
"""
import os
import matplotlib
from matplotlib import style
matplotlib.use('Agg')
import numpy as np
from utils.my_color import color_line as col
from utils.plot_misc import plot_evaluation, plot_trajectory
co = col()


def ll():
    l1 = ['ll/105', '-', co[1], 'll-105']
    l2 = ['ll/nb', '-', co[1], 'll-nb']
    l3 = ['ll/130', '-', co[0], 'll-130']
    return l1, l2, l3


def generate_list(dir_net, dir_name, epoch_list):
    l_dict = dict()
    for i, epoch in enumerate(epoch_list):
        l_dict[i] = [dir_net+'/{:s}_model-{:d}'.format(dir_name, epoch), '-', 'epoch-{:d}'.format(epoch)]
    return l_dict


def regroup_for_plot(items):
    files = []
    fmts = []
    legends = []
    for item in items:
        files.append(item[0])
        fmts.append(item[1])
        legends.append(item[2])
    return files, fmts, legends


class MyExperiments(object):
    def __init__(self):
        self.dir0 = '/home/jiange/dl/project'
        self.dir1_tf = self.dir0 + '/tf-cnn-vo/test/cnn-vo'                     # cnn-vo-tf
        self.dir2_tf = self.dir0 + '/tf-cnn-vo/test/cnn-vo-cons'                # cnn-vo-cons-tf
        self.dir3_tf = self.dir0 + '/tf-cnn-lstm-vo/test/cnn-lstm-vo'           # cnn-lstm-vo-tf
        self.dir4_tf = self.dir0 + '/tf-cnn-lstm-vo/test/cnn-lstm-vo-cons'      # cnn-lstm-vo-cons-tf
        self.dir1_pt = self.dir0 + '/pytorch-deepvo/test/cnn-vo'
        self.dir2_pt = self.dir0 + '/pytorch-deepvo/test/cnn-vo-cons'           # cnn-vo-cons-pt
        self.dir3_pt = self.dir0 + '/pytorch-deepvo/test/cnn-lstm-vo'           # cnn-lstm-vo-pt
        self.dir4_pt = self.dir0 + '/pytorch-cnn-lstm-vo/test/cnn-lstm-vo-cons'  # cnn-lstm-vo-cons-pt

        # baseline
        self.gt = [self.dir0 + '/tf-cnn-vo/dataset/ground-truth', '--', 'Ground truth']
        self.b1 = ['viso2-m', '-', 'VISO2-M']
        self.b2 = ['viso2-s', '-', 'VISO2-S']

        # experiment1
        self.e1_1 = [self.dir1_tf, '20171124_50_restore', [80]]  # ==2nd==
        self.e1_2 = [self.dir1_pt, '20171230', [90]]
        self.e1_3 = [self.dir1_pt, '20180101', [120]]           # it0 ==1st==
        self.e1_4 = [self.dir1_pt, '20180101_tb', [90]]
        self.e1_5 = [self.dir1_pt, '20180102_iks', [120]]

        # experiment2
        self.e2_1 = [self.dir2_tf, '1130', [75]]
        self.e2_2 = [self.dir2_tf, '20171222', [25]]  # ==1st-old==
        self.e2_3 = [self.dir2_pt, '20171209_10', [150]]
        self.e2_4 = [self.dir2_pt, '20171211_10', [150]]
        self.e2_5 = [self.dir2_pt, '20171214_10', [90]]
        self.e2_6 = [self.dir2_pt, '20171217', [80]]
        self.e2_7 = [self.dir2_pt, '20171218', [120]]
        self.e2_8 = [self.dir2_pt, '20180103', [70]]            # fine-tuning it0+it0r
        self.e2_9 = [self.dir2_pt, '20180103_triple', [110]]    # fine-tuning it0+it0r+it1 ==1st of all==
        self.e2_10 = [self.dir2_pt, '20180104', [160]]   # fine-tuning it0+it1 160

        self.e2_11 = [self.dir2_pt, '20180106_triple', [30, 40, 50, 60, 70]]  # scratch it0+it0r+it1 30 or 50 tested
        self.e2_12 = [self.dir2_pt, '20180106', [70]]  # scratch it0+it1 70 tested

        # experiment3
        self.e3_1 = [self.dir3_tf, '20171224_1', [75]]  # ==1st==
        self.e3_2 = [self.dir3_tf, '6499_20171224', [45]]
        self.e3_3 = [self.dir3_pt, '20171219', [110]]
        self.e3_4 = [self.dir3_pt, '20180104', [60]]  # bad

        self.e3_5 = [self.dir3_pt, '20180107', []]  # to be tested

        # experiment4
        self.e4_1 = [self.dir4_tf, '6499_20171225', [5]]  # [5, 20, 25, 30, 35, 40] (best of M4)
        self.e4_2 = [self.dir4_pt, '20171226', [110]]  # [80 ... 150]]
        self.e4_3 = [self.dir4_pt, '6499_20171226', [80, 120, 150]]  # [35, 45, 55, 60, 70, 80, 90, 100, 110]  80

    def list_now(self):
        _dir_save = 'evaluation/now'
        l1 = generate_list(*self.e2_9)
        l2 = generate_list(*self.e2_12)
        _methods = [self.gt, l1[0], l2[0]]
        _colors = ['k', co[1], co[0]]
        return _methods, _colors, _dir_save

    def list_cnn_cons_compare(self):
        _dir_save = 'evaluation/cnn-cons-20180104'
        l1 = generate_list(*self.e1_3)
        l2 = generate_list(*self.e2_8)
        l3 = generate_list(*self.e2_10)
        l4 = generate_list(*self.e2_9)
        _methods = [self.gt, l1[0], l2[0], l3[0], l4[0]]
        _colors = ['k', co[3], co[2], co[1], co[0]]
        _methods[1][2] = 'it0-epoch-120'
        _methods[2][2] = 'it0-it0r-epoch-70'
        _methods[3][2] = 'it0-it1-epoch-40'
        _methods[4][2] = 'it0-it0r-it1-epoch-110'
        return _methods, _colors, _dir_save

    def list_compare(self):
        _dir_save = 'evaluation/cnn-compare'
        l1 = generate_list(*self.e2_1)
        l2 = generate_list(*self.e2_2)
        _methods = [self.gt, l1[0], l1[1], l1[2], l2[0]]
        _colors = ['k', co[3], co[2], co[1], co[0]]
        _methods[4][2] = 'epoch-100'
        return _methods, _colors, _dir_save

    def list_cnn(self):
        _dir_save = 'evaluation/cnn-vo'
        l1 = generate_list(*self.e1_1)
        l2 = generate_list(*self.e2_2)
        _methods = [self.gt, self.b1, self.b2, l1[0], l2[0]]
        _colors = ['k', co[3], co[2], co[1], co[0]]
        _methods[3][2] = 'CNN-VO'
        _methods[4][2] = 'CNN-VO-cons'
        return _methods, _colors, _dir_save

    def list_lstm(self):
        _dir_save = 'evaluation/lstm-vo'
        l1 = generate_list(*self.e1_1)
        l2 = generate_list(*self.e2_1)
        l3 = generate_list(*self.e3_1)
        l4 = generate_list(*self.e4_1)
        _methods = [self.gt, self.b1, self.b2, l1[0], l2[2], l3[0], l4[0]]
        _colors = ['k', co[3], co[2], co[1], co[0], co[4], co[5]]
        _methods[3][2] = 'CNN-VO'
        _methods[4][2] = 'CNN-VO-cons'
        _methods[5][2] = 'CNN-LSTM-VO'
        _methods[6][2] = 'CNN-LSTM-VO-cons'
        return _methods, _colors, _dir_save


if __name__ == '__main__':
    my_plot = MyExperiments()
    methods, colors, dir_save = my_plot.list_now()

    if not os.path.exists(dir_save):
        os.makedirs(dir_save)
    file_list, fmt_list, legend_list = regroup_for_plot(methods)
    print('Plot evaluation...')
    plot_evaluation(file_list, fmt_list, colors, legend_list, dir_save)
    for sequence in np.arange(11):  # [3, 4, 5, 6, 7, 10]:
        style.use("ggplot")
        plot_trajectory(sequence, file_list, fmt_list, colors, legend_list, dir_save)

