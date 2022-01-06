# MLP
import csv
from itertools import islice
import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, train_test_split
import pandas as pd
from sklearn.utils import shuffle
from time import sleep
from sklearn import metrics
from base import GG2ee
from sklearn.model_selection import KFold

# def bit2attr(bitstr) -> list:
#     attr_vec = list()
#     for i in range(len(bitstr)):
#         attr_vec.append(int(bitstr[i]))
#     return attr_vec

def mean_relative_error(y_pred, y_test):
    assert len(y_pred) == len(y_test)
    mre = 0.0
    for i in range(len(y_pred)):
        mre = mre + abs((y_pred[i] - y_test[i]) / y_test[i])
    mre = mre * 100/ len(y_pred)
    return mre

# Large_MRE_points = pd.DataFrame()
# Large_MRE_X = []
# Large_MRE_y_test = []
# Large_MRE_y_pred = []
# Large_MRE = []

'''
1) 数据预处理
'''
# filepath = 'data/fp/sjn/R+B+Cmorgan_fp1202.csv'
filepath = 'train_Data/CV0126_train.csv'

data = pd.read_csv(filepath, encoding='gb18030').astype('float')
print(data.shape)
data = data.dropna()

print(data.shape)
data = shuffle(data)

data_x_df = pd.DataFrame(data.iloc[:, :-1])
data_y_df = pd.DataFrame(data.iloc[:, -1])

# 归一化
min_max_scaler_X = MinMaxScaler()
min_max_scaler_X.fit(data_x_df)
x_trans1 = min_max_scaler_X.transform(data_x_df)

min_max_scaler_y = MinMaxScaler()
min_max_scaler_y.fit(data_y_df)
y_trans1 = min_max_scaler_y.transform(data_y_df)

'''
3) 构建模型
'''

from keras.layers import MaxPooling1D, Conv1D, Dense, Flatten, Dropout, Activation
from keras import models
from keras.optimizers import Adam, RMSprop, SGD

def buildModel():
    model = models.Sequential()

    l17 = Dense(1024)
    l18 = Activation('relu')
    l19 = Dropout(rate=0)

    l1 = Dense(512)
    l2 = Activation('relu')
    l3 = Dropout(rate=0)

    l14 = Dense(256)
    l15 = Activation('relu')
    l16 = Dropout(rate=0)

    l4 = Dense(128)
    l5 = Activation('relu')
    l6 = Dropout(rate=0)

    l11 = Dense(64)
    l12 = Activation('relu')
    l13 = Dropout(rate=0)

    l7 = Dense(32)
    l8 = Activation('relu')
    l9 = Dropout(rate=0)

    l10 = Dense(1)

    layers = [
            l1, l2, l3,
            l4, l5, l6,
            l7, l8, l9,
            l10
    ]
    for i in range(len(layers)):
        model.add(layers[i])

    adam = Adam(lr=1e-3)
    model.compile(optimizer=adam, loss='logcosh', metrics=['mae'])

    return model

'''
4) 训练模型
'''
from sklearn import metrics

# n_split = 10
mlp_scores = []
MAEs = []
out_MAEs = []

in_y_test = []
in_y_pred = []
out_y_test = []
out_y_pred = []

in_y_test_ee = []
in_y_pred_ee = []

in_y_train_real = []
in_y_train_pred = []

for i in range(10):
    kf = KFold(n_splits=10, shuffle=True, random_state=i)
    for train_index, test_index in kf.split(x_trans1):
        X_train = x_trans1[train_index, :]
        y_train = y_trans1[train_index, :]
        X_test = x_trans1[test_index, :]
        y_test = y_trans1[test_index, :]

        print(X_train.shape, y_train.shape)
        print(X_test.shape, y_test.shape)

        model_mlp = buildModel()
        model_mlp.fit(X_train, y_train, epochs=120, validation_data=(X_test, y_test), verbose=1)

        print(model_mlp.summary())
        x1 = x_trans1
        y = y_trans1

        result = model_mlp.predict(X_test)

        y_test = np.reshape(y_test, (-1, 1))
        y_test = min_max_scaler_y.inverse_transform(y_test)
        # print('Result shape: ', result.shape)
        result = result.reshape(-1, 1)
        result = min_max_scaler_y.inverse_transform(result)

        mae = metrics.mean_absolute_error(y_test, result)
        MAEs.append(mae)

print('MRE', MAEs)
print('avg MAE', sum(MAEs) / len(MAEs))
print('max MAE', max(MAEs))
print('min MAE', min(MAEs))
