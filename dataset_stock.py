import pandas as pd
import numpy as np
import os
import sys
import time
import logging
from logging.handlers import RotatingFileHandler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from config import Config


class Dataset:
    def __init__(self, config):
        self.config = config
        self.data, self.data_column_name = self.read_data()

        self.data_num = self.data.shape[0]
        print("all data number:",self.data_num )
        self.train_num = int(self.data_num * self.config.train_data_rate)
        print("train data number:",self.train_num)

        self.mean = np.mean(self.data, axis=0)              # 数据的均值和方差
        self.std = np.std(self.data, axis=0)
        self.norm_data = (self.data - self.mean)/self.std   # 归一化，去量纲

        self.start_num_in_test = 0      # 测试集中前几天的数据会被删掉，因为它不够一个time_step

    def read_data(self):                # 读取初始数据


        init_data = pd.read_csv(self.config.train_data_path)
        init_data['datetime'] = pd.to_datetime(init_data['datetime'])
        #根据时间排序dataframe，然后查看前三条记录。
        init_data=init_data.sort_values(by=['datetime'])#
        init_data = init_data.reset_index(drop=True)#重建索引
        print(init_data.head(n=3))
        print(init_data.tail(n=3))
        #选择数据
        init_data = init_data[(init_data['datetime'] > '2012-10-01') &
                (init_data['datetime'] < '2018-10-31')]#.sort_values(by=['datetime'])

        print(init_data.head(n=3))

        init_data=init_data[init_data.columns[self.config.feature_columns]]#根据索引保留列

        print(init_data.head(n=3))

        return init_data.values, init_data.columns.tolist()     # .columns.tolist() 是获取列名

    def get_train_and_valid_data(self):
        feature_data = self.norm_data[:self.train_num]
        label_data = self.norm_data[self.config.predict_future_day : self.config.predict_future_day + self.train_num,
                                    self.config.label_in_feature_index]    # 将延后几天的数据作为label

        label_data1 = self.data[self.config.predict_future_day : self.config.predict_future_day + self.train_num,
                                    self.config.label_in_feature_index]    # 将延后几天的数据作为label



        # 每time_step行数据会作为一个样本，比如：1-50行，2-51行
        train_x = [feature_data[i:i+self.config.time_step] for i in range(self.train_num-self.config.time_step)]
        train_y = [label_data[i:i+self.config.time_step] for i in range(self.train_num-self.config.time_step)]


        train_x, train_y = np.array(train_x), np.array(train_y)
        #print(train_x.shape)

        train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=self.config.valid_data_rate,
                                                              random_state=self.config.random_seed,
                                                              shuffle=False)   # 划分训练和验证集
        return train_x, valid_x, train_y, valid_y

    def get_test_data(self, return_label_data=False):
        feature_data = self.norm_data[self.train_num:]
        sample_interval = min(feature_data.shape[0], self.config.time_step)     # 防止time_step大于测试集数量
        self.start_num_in_test = feature_data.shape[0] % sample_interval  # 这些天的数据不够一个sample_interval
        time_step_size = feature_data.shape[0] // sample_interval

        # 每time_step行数据会作为一个样本，比如：1-50行，2-51行
        test_x = [feature_data[self.start_num_in_test+i*sample_interval : self.start_num_in_test+(i+1)*sample_interval]
                   for i in range(time_step_size)]
        if return_label_data:       # 实际应用中的测试集是没有label数据的
            label_data = self.norm_data[self.train_num + self.start_num_in_test:, self.config.label_in_feature_index]
            return np.array(test_x), label_data
        return np.array(test_x)


np.random.seed(Config.random_seed)  # 设置随机种子，保证可复现
data_g = Dataset(Config)