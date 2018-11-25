#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Linjian Zhang
Email: linjian93@foxmail.com
Create Time: 2017-12-07 23:01:32
Program: 
Description: 
"""
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from numpy import mat
from tqdm import tqdm
import math


def cal_absolute_from_relative(xyz_euler):
    """
    根据相对位姿计算绝对位姿
    :param xyz_euler: 6-d vector [x y z theta_x theta_y theta_z]
    :return: 12-d vector
    """
    xyz_euler = np.array(xyz_euler)
    pose_absolute = []  # 12-d
    t1 = mat(np.eye(4))
    pose_absolute.extend([np.array(t1[0: 3, :]).reshape([-1])])
    for i in tqdm(range(len(xyz_euler))):
        x12 = xyz_euler[i, 0]
        y12 = xyz_euler[i, 1]
        z12 = xyz_euler[i, 2]
        theta1 = xyz_euler[i, 3] / 180 * np.pi
        theta2 = xyz_euler[i, 4] / 180 * np.pi
        theta3 = xyz_euler[i, 5] / 180 * np.pi
        tx = mat([[1, 0, 0], [0, math.cos(theta1), -math.sin(theta1)], [0, math.sin(theta1), math.cos(theta1)]])
        ty = mat([[math.cos(theta2), 0, math.sin(theta2)], [0, 1, 0], [-math.sin(theta2), 0, math.cos(theta2)]])
        tz = mat([[math.cos(theta3), -math.sin(theta3), 0], [math.sin(theta3), math.cos(theta3), 0], [0, 0, 1]])
        tr = tz * ty * tx
        t12 = np.row_stack((np.column_stack((tr, [[x12], [y12], [z12]])), [0, 0, 0, 1]))
        t2 = t1 * t12
        pose_absolute.extend([np.array(t2[0: 3, :]).reshape([-1])])
        t1 = t2

    return pose_absolute


def plot_from_pose(seq, dir_save, pose_abs, epoch=None, args=None):
    """
    训练和测试时画图的legend、命名、保存位置不一样
    """
    plt.close('all')
    style.use("ggplot")
    pose_gt = np.loadtxt('dataset/ground-truth/{:02d}.txt'.format(seq))
    pose_pre = np.array(pose_abs)  # [image_numbers, 6]
    plt.plot(pose_gt[:, 3], pose_gt[:, 11], '--', c='k', lw=1.5, label='Ground truth')
    if args.phase == 'Train':
        plt.plot(pose_pre[:, 3], pose_pre[:, 11], '-', c='r', lw=1.5, label='model-{:d}'.format(epoch))
    else:
        plt.plot(pose_pre[:, 3], pose_pre[:, 11], '-', c='r', lw=1.5, label=args.model_restore)

    plt.title('Sequence {:02d}'.format(seq))
    plt.xlabel('X', fontsize=12)
    plt.ylabel('Z', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    if args.phase == 'Train':
        plt.savefig(dir_save+'/{:d}-epoch-{:d}.png'.format(seq, epoch))
    else:
        plt.savefig(dir_save + '/{:d}.png'.format(seq))


def main():
    from tensorboardX import SummaryWriter
    writer = SummaryWriter('./img')
    plt.figure()
    plt.plot([1, 2])
    plt.title("test")
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    writer.add_image('Image', buf)


if '__name__' == '__main__':
    main()
