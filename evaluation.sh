#########################################################################
# File Name: evaluation.sh
# Author: Linjian Zhang
# Mail: linjian93@foxmail.com
# Created Time: 2018年01月 2日 15:50:19
#########################################################################
#!/bin/bash

for i in 70
do
    # cnn-vo-cons
    # 40 80 90 / 120 140 160 180
    # /home/jiange/dl/project/tf-cnn-vo/evaluation/cpp/test /home/jiange/dl/project/pytorch-deepvo/test/cnn-vo-cons/20180104_model-$i

    # 30 40 50 60 / 70
    /home/jiange/dl/project/tf-cnn-vo/evaluation/cpp/test /home/jiange/dl/project/pytorch-deepvo/test/cnn-vo-cons/20180106_model-$i

    # cnn-lstm-vo
    # 50 60 80 100 120 140 160 180 200
    # /home/jiange/dl/project/tf-cnn-vo/evaluation/cpp/test /home/jiange/dl/project/pytorch-deepvo/test/cnn-lstm-vo/20180104_model-$i
done

# nohup sh evaluation.sh > nohup/evaluation.log 2>&1 &