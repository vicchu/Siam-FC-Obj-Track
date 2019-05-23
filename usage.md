# Usage 运行说明

## 1. Requirements 运行要求
My running environment is listed as below.\
我的运行环境如下。
* python 3.7
* pytorch 1.0.1
* numpy 1.16.2
* opencv-python 4.1.0.25
* matplotlib 3.0.3
* pillow 5.4.1
* yaml 0.1.7
* [GOT-10k Python Toolkit](http://got-10k.aitestunion.com/)
* [Gaft](https://github.com/PytLab/gaft) (optional 可选)

## 2. Dataset 数据集
Please download these dataset and unzip them according to their introduction.\
请下载以下数据集并解压。
* [ILSVRC 2015-VID](http://image-net.org/challenges/LSVRC/2015/) (for train 用于训练)
    >You can also download it from [here](http://bvisionweb1.cs.unc.edu/ilsvrc2015/ILSVRC2015_VID.tar.gz).\
    你可以直接从[这里](http://bvisionweb1.cs.unc.edu/ilsvrc2015/ILSVRC2015_VID.tar.gz)下载。
* [OTB](http://cvlab.hanyang.ac.kr/tracker_benchmark/index.html) (for test 用于测试)
* [GOT-10k](http://got-10k.aitestunion.com/) (for test 用于测试)

## 3. Download Model 下载模型
Please download the model from [here](https://pan.baidu.com/s/1WYWzvm9jcrbL_s-OIsD1rA). Password is `2mc4`\
请下载[模型](https://pan.baidu.com/s/1WYWzvm9jcrbL_s-OIsD1rA)。提取码：`2mc4`

## 4. Set Parameters 设置参数
* ### For Trainning 训练
  1. Set the proper paths and parameters in [para_set.yaml](./para_set.yaml).\
   在[para_set.yaml](./para_set.yaml)中设置合适的路径和参数。
        
        * `data_main_path`:

            Path to the `ILSVRC 2015-VID` dataset. it should contain `Annotations`, `Data` and `ImageSets`。\
            `ILSVRC 2015-VID`数据集的路径，其下应该含有文件夹
            `Annotations`、`Data` 和 `ImageSets`。
        * `data_temp_path`:

            Path to save tempory data. It should larger than 90GB.\
            存放缓存数据的路径。应有90GB以上空间。
        * `log_path`:

            Path to save log.
            日志存放路径。
        * `checkpoint_path`:

            Path to save checkpoint file.
            存放训练模型的路径。
        * `init_lr`:

            Initial learning rate.
            初始学习率。
        * `momentum`:

            Momentum.
            动量。
        * `weight_decay`:
            
            Weight decay.
            权重衰减。
        * `gamma`:

            learning rate in i-th epoch = `gamma` x learning rate in (i-1)th epoch.\
            第i轮学习率 = `gamma` x 第i-1轮学习率。
        * `batch_size`:

            Batch size. 批大小。
        * `iter_num`:

            Number of iterations in each epoch.\
            每轮的迭代次数。
        * `epoch`:

            Number of epoches.
            迭代轮数。
        * `check_interval`:

            Save a checkpoint file every `check_interval` epoches.\
            每过`check_interval`轮保存一次训练的模型。
        * `num_workers`:
  
            Number of workers used in multi-process.\
            多线程核数。
        * `statics_proportion`:

            Proportion of the samples used to compute statistics information.\
            用于计算统计信息的样本比例。
        * `statics_frame_num`:

            Number of frames sampled from each video to compute the statistics information.\
            计算统计信息时每个视频采样的帧数。
        
        * `rgb_noise`:

            Whether to introduce noise.\
            是否引入噪声。
        * `gray_trans`:

            Whether to transform the color videos into gray videos.
            是否将彩色视频转化为灰度视频。
        * `gray_proportion`:
  
            Proportion of the color videos transfomed into gray videos.\
            彩色视频转化为灰度视频的比例。
        * `flip_trans`:

            Whether to flip horizontally.\
            是否水平翻转。
        * `flip_proportion`:

            Proportion of the videos flipped horizontally.\
            水平翻转视频的比例。
        * `stretch_trans`:

            Whether to stretch.\
            是否拉伸视频。
        * `stretch_max`:

            Max size (in pixel) to stretch.\
            最大拉伸量（像素单位）。
        * `stretch_delta`:

            Range of the proportion when stretching.\
            拉伸比例范围。
        * `rPos`, `rNeg`:

            Please refer the paper for details.
            详见论文。
        * `pair_frame_range`:

            Max time distance of two frame in a training pair randomly sampled.\
            随机采样的训练帧对的最大时间距离。
        * `loss_report_iter`:

            Report the loss every `loss_report_iter` iterations.\
            每迭代`loss_report_iter`次报告一次损失值。
    1. Save your changes. Remember to make the directories according to your set.\
    保存你的更改。记得根据你的设置建立相应文件夹。

* ### For Test 测试
    1. Set the proper paths and parameters in [arg_test.yaml](track_eval\arg_test.yaml).\
    在[arg_test.yaml](track_eval\arg_test.yaml)中设置合适的路径和参数。

        * `model_path`:

            Path to the model file.\
            模型文件路径。
        * `data_root`:

            Path to dataset OTB and GOT-10k. It should contain `OTB` and `got10k`. `OTB` contains `Basketball` and other directories. `got10k` contains `train`, `val` and `test`.\
            数据集OTB和GOT-10k的路径。其下应含文件夹`OTB`和`got10k`。其中`OTB`含有`Basketball`等文件夹。`got10k`含有`train`、`val`和`test`。
        * `report_dir`:

            Path to save results.\
            存放结果的路径。
        * `name_tracker`:

            Name of the tracker. It should be changed after every time of test.\
            跟踪器名字。每次测试后都应更换。
        * `scale_step`:

            Step between adjacent scales. Please refer the paper for details.\
            相邻尺度的步长。详见论文。
        * `scale_penalty`:

            Penalty for the scale change. Please refer the paper for details.\
            尺度变化惩罚。详见论文。
        * `scale_lr`:

            Learning rate of the scale. Please refer the paper for details.\
            尺度学习率。详见论文。
        * `hann_weight`:

            Weight of the hanning window. Please refer the paper for details.\
            汉宁窗权重。详见论文。
        
        * `scale_num`:

            Number of different scales in the pyramid.\
            金字塔尺度数目。
        * `interpolation_beta`:

            Enlarger the feature map by `interpolation_beta` times.\
            将特征图放大`interpolation_beta`倍。
        * `s_x_limit_beta`:

            Max scale = `s_x_limit_beta`, min scale = `1 / s_x_limit_beta`.\
            最大尺度 = `s_x_limit_beta`，最小尺度 = `1 / s_x_limit_beta`。
    2. Save your changes. Remember to make the directories according to your set.\
    保存你的更改。记得根据你的设置建立相应文件夹。


## 5. Run 运行
* ### For trainning 训练
    1. Execute 执行
        ```bash
        cd ${YOUR_WORK_DIR}
        python ./train/train_net.py
        ```
    2. Go to `checkpoint_path` you set in [para_set.yaml](./para_set.yaml), there will be checkpoint files and `loss_fig.pdf` which plots the curve of tranning loss.\
    前往你在[para_set.yaml](./para_set.yaml)中设置的`checkpoint_path`路径，里面存有训练的模型和绘制了训练损失曲线的`loss_fig.pdf`。
    3. Go to `log_path` you set in [para_set.yaml](./para_set.yaml), there will be log files.\
    前往你在[para_set.yaml](./para_set.yaml)中设置的`log_path`路径，里面存有日志文件。
* ### For test 测试
    1. Execute 执行
        ```bash
        cd ${YOUR_WORK_DIR}
        python ./track_eval/test_net.py
        ```
    2. Go to `report_dir` you set in [arg_test.yaml](./track_eval/arg_test.yaml), there will be report files which plots the curve for test.\
    前往你在[arg_test.yaml](./track_eval/arg_test.yaml)中设置的`report_dir`路径，里面存有测试报告文件。

## 6. Tune Parameters 调参
* [Gaft](https://github.com/PytLab/gaft) must be installed to tune parameters. Then set all things well as mentioned in [Set Parameters](#4-set-parameters-%E8%AE%BE%E7%BD%AE%E5%8F%82%E6%95%B0).\
  调参必须安装好[Gaft](https://github.com/PytLab/gaft)。然后按照[设置参数](#4-set-parameters-%E8%AE%BE%E7%BD%AE%E5%8F%82%E6%95%B0)的说明设置好。
* Execute the command:\
  执行命令：
  ```bash
  cd ${YOUR_WORK_DIR}
  python ./track_eval/tune_para.py
  ```
    >It takes very much time (maybe half a year?) so I haven't finished it yet. Good luck.\
    因为它非常耗时（大概半年？）所以目前我从未运行完它。祝你好运。

## 7. Demo 示例
* Set all things well as mentioned in [Set Parameters](#4-set-parameters-%E8%AE%BE%E7%BD%AE%E5%8F%82%E6%95%B0).\
  按照[设置参数](#4-set-parameters-%E8%AE%BE%E7%BD%AE%E5%8F%82%E6%95%B0)的说明设置好。
* Execute the command:\
  执行命令：
  ```bash
  cd ${YOUR_WORK_DIR}
  python ./track_eval/test_demo.py
  ```
  You will see several examples in which the object is tracked successfully or unsuccessfully. You can DIY the code.\
  你会看到几个跟踪成功或失败的例子。你可以自己修改代码。