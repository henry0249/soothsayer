# -*- coding: UTF-8 -*-
"""

"""

import pandas as pd
import numpy as np
import os

import time
import logging
from logging.handlers import RotatingFileHandler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from config import Config
from trainner import train
from evaluator import predict
from utils import load_logger
from utils import draw
#from dataset import Dataset
from dataset_stock import Dataset

from model.net_LSTM import Net_LSTM
from model.net_BidirectionalLSTM import Net_BidirectionalLSTM



def main(config):
    logger = load_logger(config)
    try:
        np.random.seed(config.random_seed)  # 可复现
        data_original = Dataset(config)
        Net = Net_LSTM

        if config.model==1:
            Net = Net_LSTM
        elif config.model == 2:
            Net=Net_BidirectionalLSTM

        if config.phase=="train":
            print("The soothsayer will train")
            train_X, valid_X, train_Y, valid_Y = data_original.get_train_and_valid_data()
            train(Net,config, logger, [train_X, train_Y, valid_X, valid_Y])

        if config.phase=="predict":
            print("The soothsayer will predict")
            test_X, test_Y = data_original.get_test_data(return_label_data=True)

            pred_result = predict(Net,config, test_X)       # 这里输出的是未还原的归一化预测数据
            draw(config, data_original, logger, pred_result)
    except Exception:
        logger.error("Run Error", exc_info=True)


if __name__=="__main__":
    import argparse


    # argparse方便于命令行下输入参数，可以根据需要增加更多
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--phase", default="predict", type=str, help="train or predict")
    parser.add_argument("-m", "--model", default=1, type=int, help="1:LSTM、2:Net_BidirectionalLSTM")
    args = parser.parse_args()


    c = Config()
    for key in dir(args):
        if not key.startswith("__"):
            setattr(c, key, getattr(args, key))   # 将属性值赋给Config


    main(c)
    #python main.py -p "train" -m 2
    #python main.py -p "predict" -m 2

