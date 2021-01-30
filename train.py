'''
# data generaring attention :
it will use `np.matmul`
so it needs to be sure that every row means nine consecutive days data
'''
import pandas as pd
import numpy as np
import math
def data():
    train_path = '~/PycharmProjects/ML2020/hw1/dataset/train.csv'
    data = pd.read_csv(train_path, encoding = 'big5')

    ############## data precessing #############
    data = data.iloc[:, 3:]
    data[data == 'NR'] = 0
    raw_data = data.to_numpy()

    ## extract Feature
    # get per month data
    month_data = {}
    for month in range(12):
        sample = np.empty([18, 480])
        for day in range(20):
            sample[:, day * 24 : (day + 1) * 24] = raw_data[18 * (20 * month + day) : 18 * (20 * month + day + 1), :]
        month_data[month] = sample
    # get everyday data
    x = np.empty([12 * 471, 18 * 9], dtype = float)
    y = np.empty([12 * 471, 1], dtype = float)
    for month in range(12):
        for day in range(20):
            for hour in range(24):
                if day == 19 and hour > 14:
                    continue
                x[month * 471 + day * 24 + hour, :] = month_data[month][:,day * 24 + hour : day * 24 + hour + 9].reshape(1, -1) #vector dim:18*9 (9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9)
                y[month * 471 + day * 24 + hour, 0] = month_data[month][9, day * 24 + hour + 9] #value
    # normalize
    mean_x = np.mean(x, axis=0)  # 18 * 9
    std_x = np.std(x, axis=0)  # 18 * 9
    for i in range(len(x)):  # 12 * 471
        for j in range(len(x[0])):  # 18 * 9
            if std_x[j] != 0:
                x[i][j] = (x[i][j] - mean_x[j]) / std_x[j]
    # x = np.concatenate((x, np.ones([x.shape[0],1])), axis=1)
    # data split
    x_train_set = x[: math.floor(len(x) * 0.8), :]
    y_train_set = y[: math.floor(len(y) * 0.8), :]
    x_validation = x[math.floor(len(x) * 0.8):, :]
    y_validation = y[math.floor(len(y) * 0.8):, :]
    return [x_train_set, y_train_set, x_validation, y_validation]