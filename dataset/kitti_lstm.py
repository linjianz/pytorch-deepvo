#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Linjian Zhang
Email: linjian93@foxmail.com
Create Time: 2018-01-02 13:51:59
Program: 
Description: 
"""

from __future__ import print_function, division
import numpy as np
from PIL import Image, ImageFile
from glob import glob
from torch.utils.data import Dataset, DataLoader
ImageFile.LOAD_TRUNCATED_IMAGES = True


def read_image(im_path):
    im_data = np.array(Image.open(im_path).resize((1280, 384))).astype(float) - [88.61, 93.70, 92.11]
    return im_data.astype(np.float32)


class KITTIDataSet(Dataset):
    """KITTI VO data set"""

    def __init__(self, dir_data, dir_label, img_pairs=2, start_interval=1, phase=None):
        self.dir_data = dir_data
        self.dir_label = dir_label
        self.img_pairs = img_pairs
        self.start_interval = start_interval
        self.phase = phase
        self.balance_idx = 4
        if self.img_pairs >= 8:
            self.balance_idx = 1

        self.l1, self.si, self.label = self.load_data()

    def load_data(self):
        """
        :return:
            l1: image path list
            si: start index of one sequence
            label: [image_number, 10, 6], relative pose
        """
        list1 = []
        si = []
        label = []
        if self.phase == 'Train':
            count = 0
            for i in [0, 1, 2, 8, 9]:
                img_list = glob(self.dir_data + '/{:02d}/image_2/*.png'.format(i))
                img_list.sort()
                list1.extend(img_list)
                label1 = np.loadtxt(self.dir_label + '/xyz-euler-relative-interval0/{:d}.txt'.format(i))  # [4540, 3]
                for j in np.arange(len(img_list)):
                    label_sample = np.zeros((self.img_pairs, 6))  # [10, 6]
                    if (j < len(img_list)-self.img_pairs-self.balance_idx) and (j % self.start_interval == 0):
                        si.append(count)
                        label_sample = label1[j: j+self.img_pairs]
                    label.append(label_sample)
                    count += 1
                #
                # img_list.sort(reverse=True)
                # list1.extend(img_list)
                # label1 = np.loadtxt(self.dir_label + '/xyz-euler-relative-reverse-interval0/{:d}.txt'.format(i))
                # label1 = list(label1)
                # label1.reverse()
                # for j in np.arange(len(img_list)):
                #     label_sample = np.zeros((self.img_pairs, 6))  # [10, 6]
                #     if (j < len(img_list)-self.img_pairs-1) and (j % self.start_interval == 0):
                #         si.append(count)
                #         label_sample = label1[j: j+self.img_pairs]
                #     label.append(label_sample)
                #     count += 1
        else:
            seq_val = 5
            img_list = glob(self.dir_data + '/{:02d}/image_2/*.png'.format(seq_val))
            img_list.sort()
            list1 = img_list
            label1 = np.loadtxt(self.dir_label + '/xyz-euler-relative-interval0/{:d}.txt'.format(seq_val))
            for j in np.arange(len(img_list)):
                label_sample = np.zeros((self.img_pairs, 6))
                if (j < len(img_list)-self.img_pairs-1) and (j % self.start_interval == 0):
                    si.append(j)
                    label_sample = label1[j: j+self.img_pairs]
                label.append(label_sample)

        return list1, si, label

    def __len__(self):
        return len(self.si)

    def __getitem__(self, idx):
        """ get one sample
        :param idx: the index of one sample, choose from range(len(self.si))
        :return: sample: {'img': size[T, 6, H, W], 'label': size[T, 6]}
        """

        idx = self.si[idx]
        img_list = []
        for img_path in self.l1[idx: idx + self.img_pairs + 1]:
            img = np.array((Image.open(img_path).resize((1280, 384))))  # - [88.61, 93.70, 92.11]
            img_list.append(img.astype(np.float32))

        sample = dict()
        sample['img1'] = []
        sample['img2'] = []
        sample['label'] = []
        for img_0, img_1 in zip(img_list[:-1], img_list[1:]):
            # sample['img'].append(np.concatenate((img_0, img_1), 2))
            sample['img1'].append(img_0)
            sample['img2'].append(img_1)
            sample['label'] = np.array(self.label[idx]).astype(np.float32)

        sample['img1'] = np.stack(sample['img1'], 0)  # list ==> TxHxWxC
        sample['img1'] = np.transpose(sample['img1'], [0, 3, 1, 2])  # TxHxWx6 ==> TxCxHxW

        sample['img2'] = np.stack(sample['img2'], 0)
        sample['img2'] = np.transpose(sample['img2'], [0, 3, 1, 2])

        return sample


def main():
    from time import time
    import math
    dir_data = '/media/csc105/Data/dataset-jiange/data_odometry_color/sequences'  # 6099
    # dir_data = '/media/Data/dataset_jiange/data_odometry_color/sequences'           # 6199
    data_set = KITTIDataSet(dir_data=dir_data,
                            dir_label='.',
                            img_pairs=2,
                            start_interval=1,
                            phase='Train')

    data_loader = DataLoader(data_set, batch_size=16, shuffle=True, num_workers=4)

    print('ip {}, si {}, Total samples {}, bs {}, Batch {}'.format(data_set.img_pairs, data_set.start_interval,
                                                                   data_set.__len__(), data_loader.batch_size,
                                                                   int(math.ceil(
                                                                       data_set.__len__() / data_loader.batch_size))))

    # tic = time()
    # for i_batch, sample_batch in enumerate(data_loader):
    #     spent = (time() - tic) / (i_batch+1)
    #     print('{:.3f}s'.format(spent), i_batch, sample_batch['img'].size(), sample_batch['img'].type(),
    #           sample_batch['label'].size())


if __name__ == '__main__':
    main()
