#########################################################################
# File Name: main.sh
# Author: Linjian Zhang
# Mail: linjian93@foxmail.com
# Created Time: 2018年01月 1日 10:38:54
#########################################################################
#!/bin/bash

for i in 170 180 190 200
do
    # cnn-vo-cons

    # 80 90 100 / 110 130 150
    # python -u main.py --server=6099 --net_architecture=cnn --phase=Test --net_restore=cnn-vo --dir_restore=20180110 --batch_size=32 --model_restore=model-$i --gpu=0
    # /home/jiange/dl/project/tf-cnn-vo/evaluation/cpp/test /home/jiange/dl/project/pytorch-deepvo/test/cnn-vo/20180110_model-$i

    # 40 65 80 90 100 110 / 120 130 / 140
    # python -u main.py --server=6499 --net_architecture=cnn --phase=Test --net_restore=cnn-vo-cons --dir_restore=20180109 --batch_size=16 --model_restore=model-$i --gpu=3
    # /home/jiange/mydocument/mycode/pytorch-deepvo/evaluation/cpp/test /home/jiange/mydocument/mycode/pytorch-deepvo/test/cnn-vo-cons/20180109_model-$i

    # cnn-lstm-vo
    # 50 55 80 90 /100 110 120 130 / 140 150 160 / 170 180 190 200
    python -u main.py --server=6499 --net_architecture=cnn-lstm --phase=Test --img_pairs=2 --net_restore=cnn-lstm-vo --dir_restore=20180114 --model_restore=model-$i --gpu=2
    /home/jiange/mydocument/mycode/pytorch-deepvo/evaluation/cpp/test /home/jiange/mydocument/mycode/pytorch-deepvo/test/cnn-lstm-vo/20180114_model-$i
done

# nohup sh test.sh > nohup/test.log 2>&1 &