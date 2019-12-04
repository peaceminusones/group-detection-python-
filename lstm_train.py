import os
import json
import time
import math
import matplotlib.pyplot as plt
from loadData import DataLoader

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

dataDirectory = "mydata/student003"

def main():
    configs = json.load(open('config.json','r'))
    # lstm模型训练结果存储地址
    if not os.path.exists(configs['model']['save_dir']):
        os.makedirs(configs['model']['save_dir'])
    
    datasets = ["mydata/student003","mydata/GVEII/GVEII","mydata/CMD/CMD/1airport1","mydata/CMD/CMD/1grand1"]
    # test_dataset = []

    data = DataLoader(datasets, configs['data']['train_test_split'])

    # 数据按照path读出来
    train_X,train_y = data.get_train_data(seq_len = configs['data']['sequence_length'])



if __name__ == "__main__":
    main()    

