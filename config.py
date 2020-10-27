import pandas as pd
import numpy as np
import os
import sys
import time
import logging
from logging.handlers import RotatingFileHandler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 假设我们有这样的原始数据集
# 'object_id','year_month_day','value','feature_1','feature_2','feature_3'
# 我们需要将未来n天的value设置为label，这里默认先设置未来1天
class Config:
    #设置哪些列是feature列
    #设置哪些列是要预测的列，可以有多列
    #预测未来n天
    #设置用多少天的数据来预测，要保证训练数据量大于它

    # 数据参数
    feature_columns = [1, 2, 3, 4,5]  # feature 都有哪些列，也就是'value','feature_1','feature_2','feature_3',feature_month,feature_day的索引
    # 时间没有算feature，可以把时间分解为月，日等独立作为feature
    label_columns = list([2])  # 要预测的列，按原数据从0开始计算, 这里的预测列只有一个values列，如果是价格类可以设置多个例如 [5,6]最低价和最高价
    label_in_feature_index = [1]  # 从feature列中找到value 的索引，
    predict_future_day = 1  # 预测未来几天

    #数据参数 股票数据使用
    feature_columns = [2,3,4,5,6,7]#feature 都有哪些列，也就是datetime	code	open	close	high	low	vol	amount	p_change 的索引
    #时间没有算feature，可以把时间分解为月，日等独立作为feature
    label_columns = list([4])  # 4 high,最高价 要预测的列，按原数据从0开始计算,如果是价格类可以设置多个例如 [4,5]最低价和最高价
    label_in_feature_index =[2] #从feature列中找到high 的索引，
    predict_future_day = 1             # 预测未来几天



    # 网络参数
    input_size = len(feature_columns)
    output_size = len(label_columns)
    #这里先使用LSTM模型，其他模型配置这里在版本迭代的时候再次处理

    hidden_size = 128           # LSTM的隐藏层大小，也是输出大小
    lstm_layers = 3             # LSTM的堆叠层数
    dropout_rate = 0.2          # dropout概率
    time_step = 50              # 这个参数很重要，是设置用前多少天的数据来预测，也是LSTM的time step数，请保证训练数据量大于它

    # 训练参数
    phase="train" # or predict
    load_model=False



    train_data_rate = 0.8      # 训练数据占总体数据比例，测试数据就是 1-train_data_rate
    valid_data_rate = 0.2     # 验证数据占训练数据比例，验证集在训练过程使用，为了做模型和参数选择

    batch_size = 64
    learning_rate = 0.001
    epoch = 500                 # 整个训练集被训练多少遍，不考虑早停的前提下
    patience = 50                # 训练多少epoch，验证集没提升就停掉
    random_seed = 1            # 随机种子，保证可复现


    # 框架参数
    model_name="model.pth"


    # 路径参数
    train_data_path = "./data/sh300.csv"
    model_save_path = "./checkpoint/"
    figure_save_path = "./figure/"
    log_save_path = "./log/"
    do_log_print_to_screen = True
    do_log_save_to_file = True                  # 是否将config和训练过程记录到log
    do_figure_save = True
    do_train_visualized = False          # 训练loss可视化，pytorch用visdom 或者tensorboardX
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)    # makedirs 递归创建目录
    if not os.path.exists(figure_save_path):
        os.mkdir(figure_save_path)
    if phase=="train" and (do_log_save_to_file or do_train_visualized):
        cur_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        log_save_path = log_save_path + cur_time + "/"
        os.makedirs(log_save_path)

    model=1#"NET_LSTM" #模型类型