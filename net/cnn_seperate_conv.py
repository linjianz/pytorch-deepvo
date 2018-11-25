#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Linjian Zhang
Email: linjian93@foxmail.com
Create Time: 2018-01-08 18:07:50
Program: 
Description: 
"""
import torch
import torch.nn as nn
from torch.nn.init import xavier_normal


def conv(batch_norm, c_in, c_out, ks=3, sd=1):
    if batch_norm:
        return nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=ks, stride=sd, padding=(ks-1)//2, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(),
        )
    else:
        return nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=ks, stride=sd, padding=(ks-1)//2, bias=True),
            nn.ReLU(),
        )


def fc(c_in, c_out, activation=False):
    if activation:
        return nn.Sequential(
            nn.Linear(c_in, c_out),
            nn.ReLU(),
        )
    else:
        return nn.Linear(c_in, c_out)


class Net(nn.Module):
    def __init__(self, batch_norm=False):
        super(Net, self).__init__()
        self.batch_norm = batch_norm
        self.conv1 = conv(self.batch_norm, 3, 64, ks=7, sd=2)
        self.conv2 = conv(self.batch_norm, 64, 128, ks=5, sd=2)
        self.conv3 = conv(self.batch_norm, 128, 256, ks=5, sd=2)
        self.conv3_1 = conv(self.batch_norm, 256, 256)
        self.conv4 = conv(self.batch_norm, 256, 512, sd=2)
        self.conv4_1 = conv(self.batch_norm, 512, 512)
        self.conv5 = conv(self.batch_norm, 512, 512, sd=2)
        self.conv5_1 = conv(self.batch_norm, 512, 512)
        self.conv6 = conv(self.batch_norm, 512, 1024, sd=2)
        self.conv6_1 = conv(self.batch_norm, 1024, 1024)
        self.pool_1 = nn.MaxPool2d(2, stride=2)
        self.dropout1 = nn.Dropout(0.5)
        self.fc_1 = fc(1024 * 3 * 10, 2048, activation=True)

        self.conv1_2 = conv(self.batch_norm, 3, 64, ks=7, sd=2)
        self.conv2_2 = conv(self.batch_norm, 64, 128, ks=5, sd=2)
        self.conv3_2 = conv(self.batch_norm, 128, 256, ks=5, sd=2)
        self.conv3_1_2 = conv(self.batch_norm, 256, 256)
        self.conv4_2 = conv(self.batch_norm, 256, 512, sd=2)
        self.conv4_1_2 = conv(self.batch_norm, 512, 512)
        self.conv5_2 = conv(self.batch_norm, 512, 512, sd=2)
        self.conv5_1_2 = conv(self.batch_norm, 512, 512)
        self.conv6_2 = conv(self.batch_norm, 512, 1024, sd=2)
        self.conv6_1_2 = conv(self.batch_norm, 1024, 1024)
        self.pool_1_2 = nn.MaxPool2d(2, stride=2)
        self.dropout1_2 = nn.Dropout(0.5)
        self.fc_1_2 = fc(1024*3*10, 2048, activation=True)

        self.dropout2 = nn.Dropout(0.5)
        self.fc_2 = fc(4096, 1024, activation=True)
        self.fc_3 = fc(1024, 128, activation=True)
        self.fc_4 = fc(128, 6)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x1, x2):
        x1 = self.conv2(self.conv1(x1))
        x1 = self.conv3_1(self.conv3(x1))
        x1 = self.conv4_1(self.conv4(x1))
        x1 = self.conv5_1(self.conv5(x1))
        x1 = self.conv6_1(self.conv6(x1))
        x1 = self.pool_1(x1)
        x1 = x1.view(x1.size(0), -1)
        x1 = self.dropout1(x1)
        x1 = self.fc_1(x1)  # NxD

        x2 = self.conv2_2(self.conv1_2(x2))
        x2 = self.conv3_1_2(self.conv3_2(x2))
        x2 = self.conv4_1_2(self.conv4_2(x2))
        x2 = self.conv5_1_2(self.conv5_2(x2))
        x2 = self.conv6_1_2(self.conv6_2(x2))
        x2 = self.pool_1_2(x2)
        x2 = x2.view(x2.size(0), -1)
        x2 = self.dropout1_2(x2)
        x2 = self.fc_1_2(x2)

        x = torch.cat((x1, x2), dim=1)
        x = self.dropout2(x)
        x = self.fc_2(x)
        x = self.fc_3(x)
        x = self.fc_4(x)

        return x

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]


def main():
    net = Net()
    print(net)


if __name__ == '__main__':
    main()
