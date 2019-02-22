# Pytorch-DeepVO

This is an Implementation of DeepVO with CNN / CNN-LSTM.

As for the experiment results, you can read my [Master's thesis](http://kns.cnki.net/KCMS/detail/detail.aspx?dbcode=CMFD&dbname=CMFD201802&filename=1018186763.nh&v=MTcyNDNMdXhZUzdEaDFUM3FUcldNMUZyQ1VSTE9mWnVkc0ZDbmdWYnJJVkYyNkZyS3dHTmJLckpFYlBJUjhlWDE=), or go to [Zhihu](https://www.zhihu.com/question/65068625/answer/256306051) for detailed discussion.

## 代码架构

- dataset

数据集处理的代码都放这

- net

网络结构的代码都放这

- utils

各种其他函数

- evaluation

测试结果处理，包括画误差曲线、画轨迹图

- main.py

主函数

- evaluation.sh

对测试结果进行评估

## 服务器端的其他数据

- 数据集

- log

训练网络时保存的loss等参数，用于tensorboard显示  

- model

保存的网络参数都放这，用于继续训练或测试

- test

测试结果都放这

## 训练样例

从头训练
```bash
$ python main.py \
--server=6499 \
--net_architecture=cnn \
--phase=Train \
--resume=No \
--net_name=cnn-vo \
--dir0=20180109 \
--gpu=0 \
```

继续训练
```bash
$ python main.py \
--server=6499 \
--net_architecture=cnn \
--phase=Train \
--resume=Yes \
--net_restore=cnn-vo \
--dir_restore=20180101 \
--model_restore=model-120 \
--net_name=cnn-vo-cons \
--dir0=20180103 \
--epoch_test=10 \
--gpu=2,3 \
```

测试
```bash
$ python main.py --server=6499 --net_architecture=cnn-lstm --phase=Test --img_pairs=2 --net_restore=cnn-lstm-vo --dir_restore=20180114 --model_restore=model-100 --gpu=2
```
