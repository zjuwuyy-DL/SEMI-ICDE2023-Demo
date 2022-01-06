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
from base import linear_regression
from sklearn.model_selection import KFold

def bit2attr(bitstr) -> list:
    attr_vec = list()
    for i in range(len(bitstr)):
        attr_vec.append(int(bitstr[i]))
    return attr_vec

def mean_relative_error(y_pred, y_test):
    assert len(y_pred) == len(y_test)
    mre = 0.0
    for i in range(len(y_pred)):
        mre = mre + abs((y_pred[i] - y_test[i]) / y_test[i])
    mre = mre * 100/ len(y_pred)
    return mre

Large_MRE_points = pd.DataFrame()
Large_MRE_X = []
Large_MRE_y_test = []
Large_MRE_y_pred = []
Large_MRE = []

'''
1) 数据预处理
'''
# filepath = 'data/fp/sjn/R+B+Cmorgan_fp1202.csv'
filepath = 'train_Data/BV0126_train.csv'

data = pd.read_csv(filepath, encoding='gb18030').astype('float')
print(data.shape)
L = set(data.index) - set(data.dropna().index)
for idx in L:
    print('Dropped ee:', data.iloc[idx,-1])
data = data.dropna()

print(data.shape)
data = shuffle(data)

data_x_df = pd.DataFrame(data.iloc[:, :-1])
data_y_df = pd.DataFrame(data.iloc[:, -1])

# 归一化
min_max_scaler_X = MinMaxScaler()
min_max_scaler_X.fit(data_x_df)
x_trans1 = min_max_scaler_X.transform(data_x_df)
x_trans1 = np.reshape(x_trans1, (x_trans1.shape[0], x_trans1.shape[1], 1))

min_max_scaler_y = MinMaxScaler()
min_max_scaler_y.fit(data_y_df)
y_trans1 = min_max_scaler_y.transform(data_y_df)
y_trans1 = np.reshape(y_trans1, (y_trans1.shape[0], 1, 1))

# test_filepath = "data/descriptor/01-03-test-2.csv"
# test_data = pd.read_csv(test_filepath, encoding='gb18030')
# print('test data: ', test_data.shape)
# test_data = test_data.dropna()
# test_data_x_df = pd.DataFrame(test_data.iloc[:, :-1])
# test_data_y_df = pd.DataFrame(test_data.iloc[:, -1])
# x_trans1_test = min_max_scaler_X.transform(test_data_x_df)
# y_trans1_test = min_max_scaler_y.transform(test_data_y_df)
# x_trans1_test = np.reshape(x_trans1_test, (x_trans1_test.shape[0], x_trans1_test.shape[1], 1))
# y_trans1_test = np.reshape(y_trans1_test, (y_trans1_test.shape[0], 1, 1))

'''
3) 构建模型
'''

from keras.layers import MaxPooling1D, Conv1D, Dense, Flatten, Dropout, Activation
from keras import models
from keras.optimizers import Adam, RMSprop, SGD
from keras.activations import relu

def buildModel():
    model = models.Sequential()

    l1 = Conv1D(6, 25, 1, activation='relu', use_bias=True)
    l2 = MaxPooling1D(2, 2)
    l3 = Conv1D(16, 25, 1, activation='relu', use_bias=True)
    l4 = MaxPooling1D(2, 2)
    l5 = Flatten()

    l9 = Dense(512)
    l10 = Activation('relu')
    l11 = Dropout(rate=0.3)

    l16 = Dense(512)
    l17 = Activation('relu')
    l18 = Dropout(rate=0)

    l6 = Dense(120)
    l7 = Activation('relu')
    l8 = Dropout(rate=0.1)

    l12 = Dense(84)
    l13 = Activation('relu')
    l14 = Dropout(rate=0.1)

    l15 = Dense(1)

    layers = [l1, l2, l3, l4, l5,
              l9, l10, l11,
              # l16, l17, l18,
              l6, l7, l8,
              l12, l13, l14,
              l15]
    for i in range(len(layers)):
        model.add(layers[i])

    adam = Adam(lr=1e-3)
    model.compile(optimizer=adam, loss='logcosh', metrics=['mae'])

    return model

def buildModel_2():
    model = models.Sequential()

    l1 = Conv1D(6, 25, 1, activation='relu', use_bias=True)
    l2 = MaxPooling1D(2, 2)
    l3 = Conv1D(16, 25, 1, activation='relu', use_bias=True)
    l4 = MaxPooling1D(2, 2)
    l5 = Flatten()

    l9 = Dense(512)
    l10 = Activation('relu')
    l11 = Dropout(rate=0.3)

    l16 = Dense(512)
    l17 = Activation('relu')
    l18 = Dropout(rate=0)

    l6 = Dense(120)
    l7 = Activation('relu')
    l8 = Dropout(rate=0.1)

    l12 = Dense(84)
    l13 = Activation('relu')
    l14 = Dropout(rate=0.1)

    l15 = Dense(1)
    l16 = Activation(lambda x: relu(x, max_value=1, alpha=0, threshold=-4))

    layers = [l1, l2, l3, l4, l5,
              l9, l10, l11,
              # l16, l17, l18,
              l6, l7, l8,
              l12, l13, l14,
              l15]
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

for i in range(1):
    # kf = KFold(n_splits=n_split, random_state=i, shuffle=True)
    # for train_in, test_in in kf.split(data_x_df):
    #     X_train = data_x_df.iloc[train_in, :]
    #     X_test = data_x_df.iloc[test_in, :]
    #     y_train = data_y_df.iloc[train_in]
    #     y_test = data_y_df.iloc[test_in]
    #     print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    kf = KFold(n_splits=10, shuffle=True, random_state=i)

    for train_index, test_index in kf.split(x_trans1):
        X_train = x_trans1[train_index, :]
        y_train = y_trans1[train_index, :]
        X_test = x_trans1[test_index, :]
        y_test = y_trans1[test_index, :]

        ## Initial: 0.2
        # X_train, X_test, y_train, y_test = train_test_split(x_trans1, y_trans1, test_size=0.1, shuffle=True, random_state=i)
        # train_test_split 随机划分 random_state, 填0或不填，每次都会不一样

        print(X_train.shape, y_train.shape)
        print(X_test.shape, y_test.shape)

        # sleep(5)

        ## Initial: 400 200 100
        model_mlp = buildModel_2()
        model_mlp.fit(X_train, y_train, epochs=120, validation_data=(X_test, y_test), verbose=1)

        print(model_mlp.summary())

        result = model_mlp.predict(X_test)

        y_test = np.reshape(y_test, (-1,1))
        y_test = min_max_scaler_y.inverse_transform(y_test)
        # print('Result shape: ', result.shape)
        result = result.reshape(-1,1)
        result = min_max_scaler_y.inverse_transform(result)

        # print(y_test.shape, result.shape)
        # print(result[:-20])
        mae = metrics.mean_absolute_error(y_test, result)
        MAEs.append(mae)
            # errstr = 'MAE = %.3f' % mae
            # plt.text(420, 750, errstr, fontsize=16)
            # plt.plot(y_test, result, 'ro')

        Large_MRE_X = [] ## Type of X_test??
        Large_MRE_y_test = []
        Large_MRE_y_pred = []
        Large_MRE = []

        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))
        X_test = min_max_scaler_X.inverse_transform(X_test)

        for idx in range(len(y_test)):
            Large_MRE.append(metrics.mean_absolute_error(result[idx], y_test[idx]))
        Large_MRE_y_test = list(y_test)
        Large_MRE_y_pred = list(result)

        temp = pd.DataFrame(X_test)
        temp = pd.concat([temp, pd.DataFrame({'Real Value': Large_MRE_y_test}), pd.DataFrame({'Predicted Value': Large_MRE_y_pred}),
                              pd.DataFrame({'MAE': Large_MRE})], axis=1)
        temp = temp.sort_values(by='MAE', ascending=False)
        temp.to_csv('Out/Large_MRE_points' + str(i) + '.csv', encoding='gb18030', index=False)

        for c in y_test:
            in_y_test.append(c[0])
        for c in result:
            in_y_pred.append(c[0])

        # in_y_test.append(y_test)
        # in_y_pred.append(result)

        T = X_test[:, -1]

        for c in GG2ee(y_test, T):
            in_y_test_ee.append(c[0])
        for c in GG2ee(result, T):
            in_y_pred_ee.append(c[0])
        # in_y_test_ee.append(GG2ee(y_test, T))
        # in_y_pred_ee.append(GG2ee(result, T))

        # # 外部验证
        # X_test = x_trans1_test
        # result = model_mlp.predict(x_trans1_test)
        #
        # y_trans1_test = np.reshape(y_trans1_test, (-1, 1))
        # y_test = min_max_scaler_y.inverse_transform(y_trans1_test)
        # result = result.reshape(-1, 1)
        # result = min_max_scaler_y.inverse_transform(result)
        #
        # mae = mean_relative_error(y_test, result)
        # out_MAEs.append(mae)
        # # errstr = 'MAE = %.3f' % mae
        # # plt.text(420, 750, errstr, fontsize=16)
        # # plt.plot(y_test, result, 'ro')
        #
        # Large_MRE_X = [] ## Type of X_test??
        # Large_MRE_y_test = []
        # Large_MRE_y_pred = []
        # Large_MRE = []
        #
        # X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))
        # X_test = min_max_scaler_X.inverse_transform(X_test)
        # # for idx in range(len(y_test)):
        # #     if mean_relative_error([result[idx]], [y_test[idx]]) > 5:
        # #         Large_MRE_X.append(X_test[idx])
        # #         Large_MRE_y_test.append(y_test[idx])
        # #         Large_MRE_y_pred.append(result[idx])
        # #         Large_MRE.append(mean_relative_error([result[idx]], [y_test[idx]]))
        #
        # for idx in range(len(y_test)):
        #     Large_MRE.append(mean_relative_error([result[idx]], [y_test[idx]])[0])
        # Large_MRE_y_test = list(y_test)
        # Large_MRE_y_pred = list(result)
        #
        # temp = pd.DataFrame(X_test)
        # temp = pd.concat([temp, pd.DataFrame({'Real Value': Large_MRE_y_test}), pd.DataFrame({'Predicted Value': Large_MRE_y_pred}),
        #                   pd.DataFrame({'MRE': Large_MRE})], axis=1)
        # temp = temp.sort_values(by='MRE', ascending=False)
        # temp.to_csv('Out/Large_MRE_out_points' + str(i) + '.csv', encoding='gb18030', index=False)
        #
        # out_y_test.append(y_test)
        # out_y_pred.append(result)

        result = model_mlp.predict(X_train)

        y_train = np.reshape(y_train, (-1, 1))
        y_train = min_max_scaler_y.inverse_transform(y_train)
        # print('Result shape: ', result.shape)
        result = result.reshape(-1, 1)
        result = min_max_scaler_y.inverse_transform(result)

        for c in y_train:
            in_y_train_real.append(c[0])
        for c in result:
            in_y_train_pred.append(c[0])

## 白+绿纯色颜色映射
from pylab import *
from matplotlib.colors import ListedColormap,LinearSegmentedColormap
clist = ['white', 'green', 'black']
newcmp = LinearSegmentedColormap.from_list('chaos',clist)

fig = plt.figure(figsize=(14, 10))
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
# plt.grid(linestyle="--")
plt.xlabel(r'Measured $\mathbf{\Delta\Delta G^{\ne}}$ (kcal/mol)', fontsize=20, weight='bold')
plt.ylabel(r'Predicted $\mathbf{\Delta\Delta G^{\ne}}$ (kcal/mol)', fontsize=20, weight='bold')
plt.yticks(size=16)
plt.xticks(size=16)
plt.plot([-3.9, 0.2], [-3.9, 0.2], ':', linewidth=1.5, color='gray')
print('MRE', MAEs)
print('avg MAE', sum(MAEs) / len(MAEs))
print('max MAE', max(MAEs))
print('min MAE', min(MAEs))

errstr_G = 'MAE=%.2f' % (sum(MAEs) / len(MAEs))
errstr_ee = 'MAE(ee)=%.2f' % (metrics.mean_absolute_error(in_y_test_ee, in_y_pred_ee))
r_square = 'R$\mathbf{^2}$=%.2f' % metrics.r2_score(in_y_test, in_y_pred)
plt.text(-3.5, -0.2, errstr_G, fontsize=20, weight='bold')
# plt.text(-3.5, -0.4, errstr_ee, fontsize=20, weight='bold')
# plt.text(-3.5, -0.4, r_square, fontsize=20, weight='bold')

cross_result = {'Real G': in_y_test, 'Predicted G': in_y_pred}
cross_result = pd.DataFrame(cross_result)
cross_result.to_csv('Out/cross_result_cnn.csv', index=False, encoding='gb18030')

hexf = plt.hexbin(in_y_test, in_y_pred, gridsize=27, extent=[-3.9, 0.2, -3.9, 0.2],
           cmap=newcmp)

print(in_y_pred)
a0, a1 = linear_regression(in_y_test, in_y_pred)
lx = [-3.9,0.2]
ly = [a0+a1*x for x in lx]
plt.plot(lx, ly, 'b', linewidth=2, color='black')
print('a0=',a0,' a1=',a1)

# xmin = np.array(in_y_test).min()
# xmax = np.array(in_y_test).max()
# ymin = np.array(in_y_pred).min()
# ymax = np.array(in_y_pred).max()
plt.axis([-3.9, 0.2, -3.9, 0.2])
ax = plt.gca()
ax.tick_params(top=True, right=True)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=16)
plt.savefig('pics/descriptor-fig-cnn.png')
plt.show()


## 训练数据图像

clist = ['white', '#1E90FF', 'black']
newcmp = LinearSegmentedColormap.from_list('chaos',clist)

fig = plt.figure(figsize=(14, 10))
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
# plt.grid(linestyle="--")
plt.xlabel(r'Measured $\mathbf{\Delta\Delta G^{\ne}}$ (kcal/mol)', fontsize=20, weight='bold')
plt.ylabel(r'Predicted $\mathbf{\Delta\Delta G^{\ne}}$ (kcal/mol)', fontsize=20, weight='bold')
plt.yticks(size=16)
plt.xticks(size=16)
plt.plot([-3.9, 0.2], [-3.9, 0.2], ':', linewidth=1.5, color='gray')

errstr_G = 'MAE=%.2f' % metrics.mean_absolute_error(in_y_train_real, in_y_train_pred)
r_square = 'R$\mathbf{^2}$=%.2f' % metrics.r2_score(in_y_train_real, in_y_train_pred)
plt.text(-3.5, -0.2, errstr_G, fontsize=20, weight='bold')
# plt.text(-3.5, -0.4, errstr_ee, fontsize=20, weight='bold')
plt.text(-3.5, -0.4, r_square, fontsize=20, weight='bold')


hexf = plt.hexbin(in_y_train_real, in_y_train_pred, gridsize=27, extent=[-3.9, 0.2, -3.9, 0.2],
           cmap=newcmp)
# xmin = np.array(in_y_test).min()
# xmax = np.array(in_y_test).max()
# ymin = np.array(in_y_pred).min()
# ymax = np.array(in_y_pred).max()
plt.axis([-3.9, 0.2, -3.9, 0.2])
ax = plt.gca()
ax.tick_params(top=True, right=True)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=16)
plt.savefig('pics/descriptor-fig-cnn-train.png')
plt.show()

cross_result = {'Real G': in_y_train_real, 'Predicted G': in_y_train_pred}
cross_result = pd.DataFrame(cross_result)
cross_result.to_csv('Out/cross_result_cnn_train.csv', index=False, encoding='gb18030')
